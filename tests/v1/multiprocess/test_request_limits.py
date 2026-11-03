# SPDX-License-Identifier: Apache-2.0
"""Real local ZMQ checks for opt-in request bounds."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable
import threading

# Third Party
import msgspec
import pytest
import zmq

# First Party
from lmcache.v1.multiprocess.custom_types import KVEventBatch
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.request_handler import HandlerType
from lmcache.v1.multiprocess.rpc import get_rpc_spec
from lmcache.v1.multiprocess.transport.request_limits import RequestLimits
from lmcache.v1.multiprocess.transport.zmq_impl.mq import MessageQueueServer
from lmcache.v1.multiprocess.transport.zmq_impl.wire import (
    decode_operation,
    encode_operation,
)


@contextmanager
def running_server(
    handler: Callable[[], str], *, blocking: bool = False
) -> Iterator[tuple[MessageQueueServer, zmq.Socket]]:
    """Run a local server with a two-request limit and a raw DEALER peer."""
    context = zmq.Context()
    address = "tcp://127.0.0.1:*"
    server = MessageQueueServer(address, context, RequestLimits(1024, 2, 2048))
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
