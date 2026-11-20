# SPDX-License-Identifier: Apache-2.0
"""Tests for Session and SessionManager."""

# Standard
from dataclasses import replace
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ipc_key_to_object_keys
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.session import Session, SessionManager
from lmcache.v1.multiprocess.token_codec import pack_token_ids, unpack_token_ids
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.periodic_thread import PeriodicThreadRegistry


@pytest.fixture
def hasher() -> TokenHasher:
    """TokenHasher with small chunk_size for testing."""
    return TokenHasher(chunk_size=4, hash_algorithm="blake3")


@pytest.fixture
def session(hasher: TokenHasher) -> Session:
    """A fresh Session instance."""
    return Session(request_id="req-1", hasher=hasher)


@pytest.fixture
def session_manager(hasher: TokenHasher) -> SessionManager:
    """A SessionManager with short TTL for testing expiry."""
    return SessionManager(hasher, ttl=0.1, cleanup_interval=None)


class TestSession:
    def test_set_tokens_replaces(self, session: Session) -> None:
        session.set_tokens(pack_token_ids([1, 2, 3]))
        assert unpack_token_ids(bytes(session.tokens)) == [1, 2, 3]
        session.set_tokens(pack_token_ids([4, 5, 6]))
        assert unpack_token_ids(bytes(session.tokens)) == [4, 5, 6]

    def test_get_hashes_basic(self, session: Session) -> None:
        """8 tokens, chunk_size=4 produces 2 hashes."""
        session.set_tokens(pack_token_ids(list(range(8))))
        hashes = session.get_hashes(0, 8)
        assert len(hashes) == 2

    def test_get_hashes_incremental(self, session: Session) -> None:
        """Calling get_hashes incrementally should produce same results."""
        tokens = list(range(12))
        session.set_tokens(pack_token_ids(tokens))

        # First call: compute chunks 0-1
        h_first = session.get_hashes(0, 8)
        assert len(h_first) == 2
        assert session.num_chunks_processed == 2

        # Second call: compute chunk 2 (chunks 0-1 already cached)
        h_second = session.get_hashes(8, 12)
        assert len(h_second) == 1
        assert session.num_chunks_processed == 3

        # Full range should match
        h_all = session.get_hashes(0, 12)
        assert h_all == h_first + h_second

    def test_get_hashes_idempotent(self, session: Session) -> None:
        """Calling get_hashes twice with same range returns same result."""
        session.set_tokens(pack_token_ids(list(range(8))))
        h1 = session.get_hashes(0, 8)
        h2 = session.get_hashes(0, 8)
        assert h1 == h2
        assert session.num_chunks_processed == 2  # not recomputed

    def test_get_hashes_matches_hasher(
        self, hasher: TokenHasher, session: Session
    ) -> None:
        """Session hashes should match standalone TokenHasher hashes."""
        tokens = list(range(8))
        session.set_tokens(pack_token_ids(tokens))
        session_hashes = session.get_hashes(0, 8)
        hasher_hashes = hasher.compute_chunk_hashes(tokens)
        assert session_hashes == hasher_hashes

    def test_rolling_state_across_calls(self, session: Session) -> None:
        """Simulates the lookup then store flow: lookup hashes all, store
        hashes a subrange. The rolling state from lookup should be reused."""
        tokens = list(range(12))
        session.set_tokens(pack_token_ids(tokens))

        # Lookup: hash everything
        all_hashes = session.get_hashes(0, 12)
        assert len(all_hashes) == 3

        # Store: request hashes for chunk 1-2 (tokens 4:12)
        # Should reuse already-computed hashes
        store_hashes = session.get_hashes(4, 12)
        assert store_hashes == all_hashes[1:]
        assert session.num_chunks_processed == 3  # no extra computation


class TestSessionManager:
    def test_get_or_create_new(self, session_manager: SessionManager) -> None:
        s = session_manager.get_or_create("req-1")
        assert s.request_id == "req-1"

    def test_get_or_create_existing(self, session_manager: SessionManager) -> None:
        s1 = session_manager.get_or_create("req-1")
        s2 = session_manager.get_or_create("req-1")
        assert s1 is s2

    def test_get_or_create_different_ids(self, session_manager: SessionManager) -> None:
        s1 = session_manager.get_or_create("req-1")
        s2 = session_manager.get_or_create("req-2")
        assert s1 is not s2

    def test_remove(self, session_manager: SessionManager) -> None:
        session_manager.get_or_create("req-1")
        removed = session_manager.remove("req-1")
        assert removed is not None
        assert removed.request_id == "req-1"
        # Should create a fresh session
        s = session_manager.get_or_create("req-1")
        assert s.num_chunks_processed == 0

    def test_remove_nonexistent(self, session_manager: SessionManager) -> None:
        """Removing a non-existent session should return None."""
        result = session_manager.remove("does-not-exist")
        assert result is None

    def test_cleanup_expired(self, session_manager: SessionManager) -> None:
        """Sessions older than TTL should be cleaned up."""
        session_manager.get_or_create("req-1")
        session_manager.get_or_create("req-2")
        # Wait for TTL to expire (ttl=0.1s)
        time.sleep(0.15)
        removed = session_manager.cleanup_expired()
        assert removed == 2
        # New session should be fresh
        s = session_manager.get_or_create("req-1")
        assert s.num_chunks_processed == 0

    def test_cleanup_keeps_fresh(self, session_manager: SessionManager) -> None:
        """Fresh sessions should not be cleaned up."""
        session_manager.get_or_create("req-1")
        removed = session_manager.cleanup_expired()
        assert removed == 0

    def test_periodic_cleanup_removes_expired_sessions(
        self, hasher: TokenHasher
    ) -> None:
        """Background cleanup should remove expired sessions automatically."""
        PeriodicThreadRegistry.reset()
        manager = SessionManager(hasher, ttl=0.05, cleanup_interval=0.01)
        try:
            manager.get_or_create("req-1")
            manager.get_or_create("req-2")

            threads = PeriodicThreadRegistry.get_instance().get_all()
            assert len(threads) == 1
            assert threads[0].is_running

            deadline = time.time() + 1.0
            while manager.active_count() != 0 and time.time() < deadline:
                time.sleep(0.01)

            assert manager.active_count() == 0
        finally:
            manager.close()
            assert PeriodicThreadRegistry.get_instance().get_all() == []
            PeriodicThreadRegistry.reset()


class TestSessionThreadSafety:
    """Verify Session is safe under concurrent access
    from multiple TP worker threads."""

    def test_concurrent_get_hashes(self, hasher: TokenHasher) -> None:
        """Multiple threads calling get_hashes on the same Session
        must produce correct hashes (no duplicates, no corruption).
        """
        session = Session(request_id="req-mt", hasher=hasher)
        tokens = list(range(20))  # 5 chunks of 4
        session.set_tokens(pack_token_ids(tokens))

        # Reference hashes computed single-threaded
        expected = hasher.compute_chunk_hashes(tokens)
        errors: list[str] = []
        barrier = threading.Barrier(8)

        def worker(tid: int) -> None:
            try:
                barrier.wait(timeout=5)
                hashes = session.get_hashes(0, 20)
                if hashes != expected:
                    errors.append(
                        "Thread %d: got %r, expected %r" % (tid, hashes, expected)
                    )
            except Exception as exc:
                errors.append("Thread %d: %s" % (tid, exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, "\n".join(errors)
        assert session.num_chunks_processed == 5

    def test_concurrent_set_and_get(self, hasher: TokenHasher) -> None:
        """set_tokens and get_hashes called concurrently must not
        corrupt internal state."""
        session = Session(request_id="req-mt2", hasher=hasher)
        tokens = list(range(8))  # 2 chunks
        session.set_tokens(pack_token_ids(tokens))
        expected = hasher.compute_chunk_hashes(tokens)
        errors: list[str] = []
        barrier = threading.Barrier(4)

        def reader(tid: int) -> None:
            try:
                barrier.wait(timeout=5)
                hashes = session.get_hashes(0, 8)
                if hashes != expected:
                    errors.append("Reader %d: mismatch" % tid)
            except Exception as exc:
                errors.append("Reader %d: %s" % (tid, exc))

        def writer() -> None:
            try:
                barrier.wait(timeout=5)
                # Re-set same tokens (idempotent)
                session.set_tokens(pack_token_ids(tokens))
            except Exception as exc:
                errors.append("Writer: %s" % exc)

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(3)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, "\n".join(errors)


class TestSessionGetHashesOptionalEnd:
    """
    Tests for Session.get_hashes with optional end (auto-align to last full chunk).
    """

    def test_get_hashes_optional_end_empty(self, session: Session) -> None:
        """get_hashes(0) should return empty list when no tokens set."""
        assert session.get_hashes(0) == []

    def test_get_hashes_optional_end_after_compute(self, session: Session) -> None:
        """get_hashes(0) should return all computed hashes."""
        session.set_tokens(pack_token_ids(list(range(12))))  # 3 chunks of 4
        all_hashes = session.get_hashes(0)
        assert len(all_hashes) == 3

    def test_get_hashes_optional_end_incremental(self, session: Session) -> None:
        """get_hashes(0) should accumulate hashes from incremental calls."""
        session.set_tokens(pack_token_ids(list(range(12))))  # 3 chunks of 4
        # First compute 2 chunks
        session.get_hashes(0, 8)
        # get_hashes(0) should now compute all 3 chunks
        assert len(session.get_hashes(0)) == 3

    def test_get_hashes_optional_end_matches_explicit(self, session: Session) -> None:
        """get_hashes(0) should return the same as get_hashes(0, aligned_len)."""
        session.set_tokens(pack_token_ids(list(range(12))))
        expected = session.get_hashes(0, 12)
        assert session.get_hashes(0) == expected

    def test_get_hashes_optional_end_truncates_partial_chunk(
        self, session: Session
    ) -> None:
        """get_hashes(0) should ignore trailing tokens that don't fill a chunk."""
        # 14 tokens with chunk_size=4 -> 3 full chunks (12 tokens), 2 leftover
        session.set_tokens(pack_token_ids(list(range(14))))
        hashes = session.get_hashes(0)
        assert len(hashes) == 3

    def test_get_hashes_optional_end_matches_hasher(
        self, hasher: TokenHasher, session: Session
    ) -> None:
        """get_hashes(0) should match standalone TokenHasher results."""
        tokens = list(range(12))
        session.set_tokens(pack_token_ids(tokens))
        session_hashes = session.get_hashes(0)
        hasher_hashes = hasher.compute_chunk_hashes(tokens)
        # Convert session hashes to bytes for comparison
        converted = [TokenHasher.hash_to_bytes(h) for h in session_hashes]
        assert converted == hasher_hashes


class TestSessionLookupIpcKey:
    """Tests for Session.lookup_ipc_key field."""

    def test_lookup_ipc_key_default_none(self, session: Session) -> None:
        """lookup_ipc_key should default to None."""
        assert session.lookup_ipc_key is None

    def test_lookup_ipc_key_set_and_get(self, session: Session) -> None:
        """lookup_ipc_key should be settable and retrievable."""
        key = IPCCacheServerKey.from_token_ids(
            model_name="test-model",
            world_size=1,
            worker_id=None,
            token_ids=list(range(8)),
            start=0,
            end=8,
            request_id="req-1",
        )
        session.lookup_ipc_key = key
        assert session.lookup_ipc_key is key
        assert session.lookup_ipc_key.model_name == "test-model"
        assert session.lookup_ipc_key.worker_id is None


class TestSessionManagerRemoveReturnsSession:
    """Tests for SessionManager.remove returning the removed session."""

    def test_remove_returns_session_with_state(self, hasher: TokenHasher) -> None:
        """remove() should return the session with all accumulated state."""
        mgr = SessionManager(hasher, ttl=600, cleanup_interval=None)
        session = mgr.get_or_create("req-1")
        session.set_tokens(pack_token_ids(list(range(8))))
        session.get_hashes(0, 8)

        key = IPCCacheServerKey.from_token_ids(
            model_name="model",
            world_size=1,
            worker_id=None,
            token_ids=list(range(8)),
            request_id="req-1",
        )
        session.lookup_ipc_key = key

        removed = mgr.remove("req-1")
        assert removed is not None
        assert removed.request_id == "req-1"
        assert len(removed.get_hashes(0)) == 2
        assert removed.lookup_ipc_key is key

    def test_remove_clears_from_manager(self, hasher: TokenHasher) -> None:
        """After remove(), the session should no longer be in the manager."""
        mgr = SessionManager(hasher, ttl=600, cleanup_interval=None)
        mgr.get_or_create("req-1")
        mgr.remove("req-1")
        assert mgr.active_count() == 0

    def test_remove_nonexistent_returns_none(self, hasher: TokenHasher) -> None:
        """remove() on a non-existent request_id should return None."""
        mgr = SessionManager(hasher, ttl=600, cleanup_interval=None)
        assert mgr.remove("no-such-id") is None


class TestSessionObjectKeys:
    @staticmethod
    def key(tokens: list[int]) -> IPCCacheServerKey:
        return IPCCacheServerKey.from_token_ids(
            model_name="model",
            world_size=2,
            worker_id=None,
            token_ids=tokens,
            start=0,
            end=len(tokens),
            request_id="req-1",
            num_kv_readers=1,
        )

    def test_extension_reuses_identities(self, session: Session) -> None:
        first = self.key(list(range(8)))
        session.set_tokens(first.token_bytes)
        old = session.get_object_keys(first, 0, 8, [0, 1])
        extended = self.key(list(range(16)))
        session.set_tokens(extended.token_bytes)
        current = session.get_object_keys(extended, 0, 16, [0, 1])
        expected = ipc_key_to_object_keys(
            extended, session.hasher.compute_chunk_hashes(list(range(16))), [0, 1]
        )
        assert current == expected
        assert all(current[g][i] is old[g][i] for g in range(2) for i in range(4))
        current[0].clear()
        assert session.get_object_keys(extended, 0, 16, [0, 1]) == expected

    @pytest.mark.parametrize("tokens", [list(range(4)), [9] * 8, list(range(6))])
    def test_replacement_invalidates_cached_prefix(
        self, session: Session, tokens: list[int]
    ) -> None:
        original = self.key(list(range(12)))
        session.set_tokens(original.token_bytes)
        session.get_object_keys(original, 0, 12, [0])
        session.set_tokens(pack_token_ids(tokens))
        extended_tokens = tokens + [42] * (12 - len(tokens))
        changed = self.key(extended_tokens)
        session.set_tokens(changed.token_bytes)
        expected_hashes = session.hasher.compute_chunk_hashes(extended_tokens)
        assert session.get_hashes(0, 12) == expected_hashes
        assert session.get_object_keys(changed, 0, 12, [0]) == ipc_key_to_object_keys(
            changed, expected_hashes, [0]
        )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("cache_salt", "other"),
            ("model_name", "other"),
            ("world_size", 3),
            ("worker_id", 0),
            ("worker_id", 1),
        ],
    )
    def test_identity_changes_do_not_alias(
        self, session: Session, field: str, value: str | int
    ) -> None:
        key = self.key(list(range(8)))
        session.set_tokens(key.token_bytes)
        original = session.get_object_keys(key, 0, 8, [0])
        changed = replace(key, **{field: value})
        expected = ipc_key_to_object_keys(
            changed, session.hasher.compute_chunk_hashes(list(range(8))), [0]
        )
        assert session.get_object_keys(changed, 0, 8, [0]) == expected
        assert expected != original
        assert session.get_object_keys(key, 0, 8, [0]) == original

    def test_group_and_range_order(self, session: Session) -> None:
        key = self.key(list(range(12)))
        session.set_tokens(key.token_bytes)
        session.get_object_keys(key, 0, 12, [0, 1])
        assert session.get_object_keys(key, 4, 8, [1, 0]) == ipc_key_to_object_keys(
            key, session.hasher.compute_chunk_hashes(list(range(12)))[1:2], [1, 0]
        )
        with pytest.raises(ValueError):
            session.get_object_keys(key, 0, 16, [0])

    def test_lookup_snapshot_survives_replacement(self, session: Session) -> None:
        first = replace(self.key(list(range(12))), start=4)
        prepared = session.prepare_lookup(first, (-1,), [0])
        second = replace(self.key([42] * 12), start=8, cache_salt="second")
        replacement = session.prepare_lookup(second, (-1,), [0])
        assert prepared.chunk_hashes == tuple(
            session.hasher.compute_chunk_hashes(list(range(12)))
        )
        assert prepared.covered_object_keys == tuple(
            ipc_key_to_object_keys(first, list(prepared.chunk_hashes[:1]), [0])[0]
        )
        assert replacement.chunk_hashes == tuple(
            session.hasher.compute_chunk_hashes([42] * 12)
        )
        assert not session.is_lookup_current(prepared.generation)
        assert session.is_lookup_current(replacement.generation)
        assert not session.record_prefetch_result(
            3, (), lookup_generation=prepared.generation
        )
