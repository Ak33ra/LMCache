# SPDX-License-Identifier: Apache-2.0
"""Real local ZMQ checks for opt-in request bounds."""

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable
import gc
import threading

# Third Party
import msgspec
import pytest
import zmq

# First Party
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    KVEventBatch,
    PrepareRetrieveResponse,
)
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.request_handler import HandlerType
from lmcache.v1.multiprocess.rpc import get_rpc_spec
from lmcache.v1.multiprocess.transport.request_limits import RequestLimits
from lmcache.v1.multiprocess.transport.zmq_impl.mq import (
    MessageQueueClient,
    MessageQueueServer,
)
from lmcache.v1.multiprocess.transport.zmq_impl.wire import (
    decode_operation,
    encode_operation,
)

TEST_LIMITS = RequestLimits(1024, 2, 2048)


@contextmanager
def running_server(
    handler: Callable[[], str],
    *,
    blocking: bool = False,
    limits: RequestLimits = TEST_LIMITS,
) -> Iterator[tuple[MessageQueueServer, zmq.Socket]]:
    """Run a bounded local server with a raw DEALER peer."""
    context = zmq.Context()
    address = "tcp://127.0.0.1:*"
    server = MessageQueueServer(address, context, limits)
    server.add_handler(
        "noop", HandlerType.BLOCKING if blocking else HandlerType.SERIALIZED, handler
    )
    if blocking:
        server.add_normal_thread_pool(["noop"], max_workers=1)
    peer = context.socket(zmq.DEALER)
    peer.setsockopt(zmq.RCVTIMEO, 3000)
    peer.connect(server.socket.getsockopt_string(zmq.LAST_ENDPOINT))
    server.start()
    try:
        yield server, peer
    finally:
        server.close()
        peer.close(linger=0)
        context.term()


def send(peer: zmq.Socket, uid: int, *payloads: bytes) -> None:
    """Send a noop request with optional malformed/oversized payloads."""
    peer.send_multipart(
        [msgspec.msgpack.encode(uid), encode_operation("noop"), *payloads]
    )


def receive(peer: zmq.Socket) -> tuple[int, str, list[bytes]]:
    """Read one reply and its operation, including explicit refusal replies."""
    uid, operation, *payloads = peer.recv_multipart()
    return msgspec.msgpack.decode(uid), decode_operation(operation), payloads


def test_running_and_queued_requests_share_limit() -> None:
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def handler() -> str:
        calls.append(1)
        entered.set()
        assert release.wait(3)
        return "ok"

    with running_server(handler, blocking=True) as (_, peer):
        try:
            send(peer, 1)
            assert entered.wait(3)
            send(peer, 2)
            send(peer, 3)
            assert receive(peer)[:2] == (3, "__request_refused__")
            assert calls == [1]
        finally:
            release.set()
        assert {receive(peer)[0], receive(peer)[0]} == {1, 2}
        send(peer, 4)
        assert receive(peer)[:2] == (4, "noop")


def test_aggregate_oversize_and_decode_failure_release_capacity() -> None:
    def handler() -> str:
        return "ok"

    with running_server(handler) as (_, peer):
        send(peer, 1, b"x" * 600, b"x" * 600)
        assert receive(peer)[:2] == (1, "__request_refused__")
        send(peer, 2, b"unexpected argument")
        assert receive(peer)[:2] == (2, "__request_refused__")
        send(peer, 3)
        assert receive(peer)[:2] == (3, "noop")


@pytest.mark.parametrize("blocking", [False, True])
def test_failure_and_oversized_response_do_not_strand_credit(blocking: bool) -> None:
    calls = 0

    def handler() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("test handler failure")
        return "x" * 2048 if calls == 2 else "ok"

    with running_server(handler, blocking=blocking) as (_, peer):
        for uid in (1, 2):
            send(peer, uid)
            assert receive(peer)[:2] == (uid, "__request_refused__")
        send(peer, 3)
        assert receive(peer)[:2] == (3, "noop")


def test_shutdown_waits_for_handler_and_is_repeatable() -> None:
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    def handler() -> str:
        entered.set()
        assert release.wait(3)
        return "ok"

    with running_server(handler, blocking=True) as (server, peer):
        send(peer, 1)
        assert entered.wait(3)

        def close() -> None:
            server.close()
            closed.set()

        thread = threading.Thread(target=close)
        thread.start()
        try:
            assert not closed.wait(0.05)
        finally:
            release.set()
        thread.join(timeout=3)
        assert closed.is_set()
        server.close()


@pytest.mark.parametrize(
    "values", [(0, 1, 1), (1024, 0, 1), (1024, 1, 0), (True, 1, 1)]
)
def test_invalid_limits(values: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        RequestLimits(*values)


def test_workspace_accounts_aggregate_copies() -> None:
    limits = RequestLimits(1024, 2, 2048)
    limits.check_workspace_bytes(2048)
    with pytest.raises(ValueError, match="workspace"):
        limits.check_workspace_bytes(2049)


@pytest.mark.parametrize(
    "staging_mib,payload_mib",
    [(1, 8), (16, 8), (32, 16), (64, 32), (128, 64), (256, 64)],
)
def test_partition_admission_scales_with_bounded_staging(
    staging_mib: int, payload_mib: int
) -> None:
    limits = RequestLimits.for_partition(staging_mib << 20)
    assert limits.max_inflight_requests == 4
    assert limits.max_payload_bytes == payload_mib << 20
    assert limits.max_workspace_bytes == 4 * payload_mib << 20
    limits.check_payload_bytes(limits.max_payload_bytes)
    limits.check_workspace_bytes(limits.max_workspace_bytes)
    with pytest.raises(ValueError, match="payload"):
        limits.check_payload_bytes(limits.max_payload_bytes + 1)
    with pytest.raises(ValueError, match="workspace"):
        limits.check_workspace_bytes(limits.max_workspace_bytes + 1)


@pytest.mark.parametrize("staging", [0, -1, True, 16.0])
def test_partition_admission_rejects_invalid_reserve(staging: int) -> None:
    with pytest.raises(ValueError, match="staging"):
        RequestLimits.for_partition(staging)


def test_large_partition_reply_and_oversize_refusal_recover_credit() -> None:
    replies = iter([32 << 20, 64 << 20, 2])

    def handler() -> str:
        return "x" * next(replies)

    with running_server(handler, limits=RequestLimits.for_partition(128 << 20)) as (
        _,
        peer,
    ):
        send(peer, 1)
        uid, operation, payloads = receive(peer)
        assert (uid, operation) == (1, "noop")
        assert msgspec.msgpack.decode(payloads[0]) == "x" * (32 << 20)
        send(peer, 2)
        assert receive(peer)[:2] == (2, "__request_refused__")
        send(peer, 3)
        assert receive(peer)[:2] == (3, "noop")


def test_stream_subscription_is_refused_before_invocation() -> None:
    def handler() -> str:
        return "ok"

    called = threading.Event()

    def subscribe(
        instance_id: int, model_name: str, cursor: int, max_events: int
    ) -> MessagingStream[KVEventBatch]:
        called.set()
        raise RuntimeError("Streaming handler must not be invoked")

    with running_server(handler) as (server, peer):
        server.add_sync_handler(get_rpc_spec("subscribe_kv_events"), subscribe)
        peer.send_multipart(
            [msgspec.msgpack.encode(1), encode_operation("subscribe_kv_events")]
            + [msgspec.msgpack.encode(item) for item in (1, "model", 0, 1)]
        )
        assert receive(peer)[:2] == (1, "__request_refused__")
        assert not called.is_set()
        send(peer, 2)
        assert receive(peer)[:2] == (2, "noop")


def test_disconnected_destination_does_not_block_next_client() -> None:
    entered = threading.Event()
    release = threading.Event()

    def handler() -> str:
        entered.set()
        assert release.wait(3)
        return "ok"

    with running_server(handler, blocking=True) as (server, peer):
        try:
            send(peer, 1)
            assert entered.wait(3)
            send(peer, 2)
            send(peer, 3)
            assert receive(peer)[:2] == (3, "__request_refused__")
            address = peer.getsockopt_string(zmq.LAST_ENDPOINT)
            peer.close(linger=0)
        finally:
            release.set()
        next_peer = server.ctx.socket(zmq.DEALER)
        next_peer.setsockopt(zmq.RCVTIMEO, 3000)
        next_peer.connect(address)
        try:
            # Each refusal is terminal, so retrying cannot grow accepted work.
            for uid in range(4, 100):
                send(next_peer, uid)
                if receive(next_peer)[1] == "noop":
                    break
            else:
                pytest.fail("Disconnected requests did not release capacity")
        finally:
            next_peer.close(linger=0)


def test_transient_response_pressure_retries_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single EAGAIN must not lose an already accepted RPC reply."""
    failed = threading.Event()
    original_send = zmq.Socket.send_multipart

    def handler() -> str:
        return "ok"

    with running_server(handler) as (server, peer):

        def send_once_blocked(
            socket: zmq.Socket, frames: list[bytes], flags: int = 0, **kwargs: Any
        ) -> None:
            if socket is server.socket and not failed.is_set():
                failed.set()
                raise zmq.Again()
            original_send(socket, frames, flags=flags, **kwargs)

        monkeypatch.setattr(zmq.Socket, "send_multipart", send_once_blocked)
        send(peer, 1)
        assert failed.wait(3)
        assert receive(peer) == (1, "noop", [msgspec.msgpack.encode("ok")])
        send(peer, 2)
        assert receive(peer)[:2] == (2, "noop")


def test_unsent_response_retains_admission_credit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsent reply and a running handler exhaust the two-request limit."""
    failed = threading.Event()
    writable = threading.Event()
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    original_send = zmq.Socket.send_multipart

    def handler() -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            assert release.wait(10)
        return "ok"

    with running_server(handler, blocking=True) as (server, peer):

        def send_when_writable(
            socket: zmq.Socket, frames: list[bytes], flags: int = 0, **kwargs: Any
        ) -> None:
            if (
                socket is server.socket
                and msgspec.msgpack.decode(frames[1]) == 1
                and not writable.is_set()
            ):
                failed.set()
                raise zmq.Again()
            original_send(socket, frames, flags=flags, **kwargs)

        monkeypatch.setattr(zmq.Socket, "send_multipart", send_when_writable)
        try:
            send(peer, 1)
            assert failed.wait(3)
            send(peer, 2)
            assert entered.wait(3)
            send(peer, 3)
            assert receive(peer)[:2] == (3, "__request_refused__")
            assert calls == 2
        finally:
            writable.set()
            release.set()
        replies = {receive(peer)[0], receive(peer)[0]}
        assert replies == {1, 2}
        send(peer, 4)
        assert receive(peer)[:2] == (4, "noop")


def test_blocked_response_allows_other_peer_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow peer must not stall another peer or server shutdown."""
    failed = threading.Event()
    closed = threading.Event()
    original_send = zmq.Socket.send_multipart

    def handler() -> str:
        return "ok"

    with running_server(handler) as (server, peer):

        def send_other_responses(
            socket: zmq.Socket, frames: list[bytes], flags: int = 0, **kwargs: Any
        ) -> None:
            if socket is server.socket and msgspec.msgpack.decode(frames[1]) == 1:
                failed.set()
                raise zmq.Again()
            original_send(socket, frames, flags=flags, **kwargs)

        monkeypatch.setattr(zmq.Socket, "send_multipart", send_other_responses)
        next_peer = server.ctx.socket(zmq.DEALER)
        next_peer.setsockopt(zmq.RCVTIMEO, 3000)
        next_peer.connect(peer.getsockopt_string(zmq.LAST_ENDPOINT))
        try:
            send(peer, 1)
            assert failed.wait(3)
            send(next_peer, 2)
            assert receive(next_peer)[:2] == (2, "noop")

            def close() -> None:
                server.close()
                closed.set()

            thread = threading.Thread(target=close)
            thread.start()
            thread.join(timeout=3)
            assert closed.is_set()
            server.close()
        finally:
            next_peer.close(linger=0)


def test_concurrent_tcp_replies_complete_without_fault_injection() -> None:
    """Concurrent callers sharing one client must receive every accepted reply."""

    def handler() -> str:
        return "ok"

    with running_server(handler) as (server, peer):
        client = MessageQueueClient(
            peer.getsockopt_string(zmq.LAST_ENDPOINT), server.ctx
        )

        def request_loop() -> None:
            for _ in range(5000):
                assert client.submit_request("noop", []).result(timeout=3) == "ok"

        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                tasks = [pool.submit(request_loop) for _ in range(2)]
                for task in tasks:
                    task.result(timeout=30)
        finally:
            client.close()


def test_large_reply_owns_bytes_after_later_replies_and_close() -> None:
    """Decoded payloads survive ZMQ frame release and socket shutdown."""
    context = zmq.Context()
    size = 256 * 1024
    server = MessageQueueServer(
        "tcp://127.0.0.1:*", context, RequestLimits(2 * size, 2, 4 * size)
    )

    def prepare(key: IPCCacheServerKey, instance_id: int) -> PrepareRetrieveResponse:
        return PrepareRetrieveResponse(success=True, data=bytes([instance_id]) * size)

    server.add_handler("prepare_retrieve", HandlerType.BLOCKING, prepare)
    server.add_normal_thread_pool(["prepare_retrieve"], max_workers=1)
    server.start()
    client = MessageQueueClient(
        server.socket.getsockopt_string(zmq.LAST_ENDPOINT), context
    )
    key = IPCCacheServerKey.from_token_ids("test", 1, 0, [1], end=1)
    responses = []
    try:
        for index in range(12):
            response = client.submit_request("prepare_retrieve", [key, index]).result(
                timeout=3
            )
            assert response.success
            assert isinstance(response.data, bytes)
            responses.append(response)
    finally:
        client.close()
        server.close()
        context.term()
    gc.collect()
    for index, response in enumerate(responses):
        assert response.data == bytes([index]) * size
