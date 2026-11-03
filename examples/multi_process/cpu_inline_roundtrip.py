# SPDX-License-Identifier: Apache-2.0
"""Verify typed inline KV bytes against a separate CPU MP server over TCP."""

# Standard
from importlib.machinery import EXTENSION_SUFFIXES
import os
import socket
import subprocess
import sys
import tempfile

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata
from lmcache.v1.multiprocess.transfer_context.inline import EngineDrivenContextInline
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
import lmcache.lmcache_native as native

CHUNK_SIZE = 8
CHUNK_COUNT = 2
SHAPE = torch.Size([2, 2, CHUNK_SIZE, 4])
MODEL = "cpu-inline-verification"
RPC_TIMEOUT = 5.0
POOL_BYTES = 64 * 1024 * 1024


def _key(
    request_id: str, token_count: int = CHUNK_SIZE * CHUNK_COUNT
) -> IPCCacheServerKey:
    """Build a deterministic token prefix with a distinct request session."""
    return IPCCacheServerKey.from_token_ids(
        model_name=MODEL,
        world_size=1,
        worker_id=0,
        token_ids=list(range(token_count)),
        end=token_count,
        request_id=request_id,
    )


def _lookup(client: RequestClient, key: IPCCacheServerKey, expected: int) -> None:
    """Check a completed prefix lookup without consuming its read reservation."""
    client.lookup(key, 1).result(timeout=RPC_TIMEOUT)
    count = client.wait_prefetch_status(key.request_id, RPC_TIMEOUT).result(
        timeout=RPC_TIMEOUT + 1
    )
    if count != expected:
        raise RuntimeError(f"{key.request_id}: expected {expected} chunks, got {count}")


def verify_inline_roundtrip(client: RequestClient) -> None:
    """Verify inline KV bytes and release the client registration.

    Args:
        client: Connected request client for an otherwise empty test server
            with eight-token chunks and the default L1-retaining store policy.
            The caller owns the client and server lifetimes.

    Returns:
        None after all checks and registration cleanup complete.

    Raises:
        RuntimeError: A lookup or transfer violates the expected contract.
        TimeoutError: An RPC exceeds its deadline.
        AssertionError: Retrieved tensor values differ from the originals.

    Note:
        Clears the test server through its public management RPC. This is
        intended for isolated verification servers, not a populated deployment.
    """
    instance_id = os.getpid()
    registration = client.register_kv_cache_engine_driven_context(
        RegisterEngineDrivenContextPayload(
            instance_id=instance_id,
            model_name=MODEL,
            world_size=1,
            block_size=4,
            num_layers=2,
            hidden_dim_size=4,
            dtype_str="float32",
            use_mla=False,
        )
    ).result(timeout=RPC_TIMEOUT)
    if registration.shm_name:
        raise RuntimeError("Expected inline transport without shared memory")
    metadata = EngineDrivenContextMetadata(
        layout_desc=MemoryLayoutDesc([SHAPE], [torch.float32]),
        block_size=4,
        use_mla=False,
    )
    context = EngineDrivenContextInline(metadata, client, RPC_TIMEOUT)
    try:
        miss = _key("initial-miss")
        _lookup(client, miss, 0)
        if context.prepare_retrieve(miss, instance_id, CHUNK_COUNT) is not None:
            raise RuntimeError("Unstored key unexpectedly returned tensors")
        client.end_session(miss.request_id).result(timeout=RPC_TIMEOUT)

        chunks = [
            torch.arange(SHAPE.numel(), dtype=torch.float32).reshape(SHAPE) + i * 1000
            for i in range(CHUNK_COUNT)
        ]
        expected = [chunk.clone() for chunk in chunks]
        store_key = _key("store")
        context.prepare_store(store_key, instance_id)
        if not context.commit_store(store_key, instance_id, chunks):
            raise RuntimeError("Typed inline store failed")
        for chunk in chunks:
            chunk.fill_(-1)
        client.end_session(store_key.request_id).result(timeout=RPC_TIMEOUT)

        for request_id in ("first-read", "repeat-read"):
            read_key = _key(request_id)
            _lookup(client, read_key, CHUNK_COUNT)
            retrieved = context.prepare_retrieve(read_key, instance_id, CHUNK_COUNT)
            if retrieved is None:
                raise RuntimeError(f"{request_id}: committed data was not readable")
            for actual, original in zip(retrieved, expected, strict=True):
                torch.testing.assert_close(actual, original, rtol=0, atol=0)
                actual.fill_(-2)
            client.commit_retrieve(read_key, instance_id).result(timeout=RPC_TIMEOUT)
            client.end_session(read_key.request_id).result(timeout=RPC_TIMEOUT)

        prefix = _key("prefix-hit", token_count=CHUNK_SIZE * 3)
        _lookup(client, prefix, CHUNK_COUNT)
        client.free_lookup_locks(prefix, 1).result(timeout=RPC_TIMEOUT)
        client.end_session(prefix.request_id).result(timeout=RPC_TIMEOUT)
        client.clear().result(timeout=RPC_TIMEOUT)
        cleared = _key("after-clear")
        _lookup(client, cleared, 0)
        client.end_session(cleared.request_id).result(timeout=RPC_TIMEOUT)
    finally:
        context.close()
        client.unregister_kv_cache_engine_driven_context(instance_id).result(
            timeout=RPC_TIMEOUT
        )


def main() -> None:
    """Start a bounded server, verify CPU bytes, and terminate the child.

    Returns:
        None after printing the successful checks.

    Raises:
        RuntimeError: CPU/native prerequisites or a protocol check fail.
        TimeoutError: An RPC exceeds its deadline.
        AssertionError: Retrieved tensors differ from the committed source.
    """
    if torch_device_type != "cpu":
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES='' for the CPU baseline")
    if not any(native.__file__.endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        raise RuntimeError("A compiled lmcache_native extension is required")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "lmcache.v1.multiprocess.server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--chunk-size",
        str(CHUNK_SIZE),
        "--supported-transfer-mode",
        "engine_driven",
        "--l1-size-gb",
        str(POOL_BYTES / (1 << 30)),
        "--no-l1-use-lazy",
        "--shm-name",
        "",
        "--eviction-policy",
        "LRU",
        "--disable-observability",
        "--disable-metrics",
    ]
    environment = dict(os.environ, LMCACHE_TRACK_USAGE="false")
    with tempfile.TemporaryFile(mode="w+") as server_log:
        server = subprocess.Popen(
            command,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            client = RequestClientFactory.create(f"tcp://127.0.0.1:{port}")
            try:
                if client.get_chunk_size().result(timeout=30) != CHUNK_SIZE:
                    raise RuntimeError(
                        "Server chunk geometry does not match the client"
                    )
                verify_inline_roundtrip(client)
            finally:
                client.close()
        except Exception:
            server_log.seek(0)
            sys.stderr.write(server_log.read())
            raise
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
                raise RuntimeError(
                    "Server did not shut down within 15 seconds"
                ) from None
        if server.returncode != 0:
            server_log.seek(0)
            sys.stderr.write(server_log.read())
            raise RuntimeError(f"Server exited with status {server.returncode}")
    print(
        "PASS: separate-process TCP typed inline roundtrip; "
        "2 chunks, shape=(2, 2, 8, 4), float32; pool=67108864 bytes"
    )
    print(
        "PASS: initial miss, source mutation, exact bytes, independent repeat read, "
        "prefix hit, lock release, clear miss, graceful server shutdown"
    )


if __name__ == "__main__":
    main()
