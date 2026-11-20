# SPDX-License-Identifier: Apache-2.0
"""
Session and SessionManager for tracking per-request state
in the multiprocess cache server.
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, overload
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey, ipc_key_to_object_keys
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.token_codec import TOKEN_STRIDE, num_packed_tokens
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    PeriodicThreadRegistry,
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class LookupObservation:
    """Consumed lookup result, with interface-origin counts per key-grid row.

    L1 counts include temporary reuse. L2 counts do not imply remote locality.
    A subsequent retrieve can still fail after the lookup completed.
    """

    hit_chunks: int
    l1_resident_keys: tuple[int, ...]
    l2_loaded_keys: tuple[int, ...]


@dataclass
class Session:
    """Tracks accumulated token IDs and computed chunk hashes for a request.

    Tokens are held in the packed form of
    :mod:`lmcache.v1.multiprocess.token_codec`, the layout that arrives on
    the wire and that blake3 hashes; only callers needing ints unpack.

    Thread-safe: all public methods are protected by an internal lock
    to allow concurrent access from multiple TP worker threads.
    """

    request_id: str
    hasher: TokenHasher
    tokens: bytes = b""
    chunk_hashes: list = field(default_factory=list)
    last_prefix_hash: Any = None
    num_chunks_processed: int = 0
    created_at: float = field(default_factory=time.time)
    lookup_ipc_key: Optional[IPCCacheServerKey] = None
    prefetch_hit_chunks: int = -1
    prefetch_locked_gids: tuple = ()
    prefetch_group_windows: tuple[int, ...] = ()
    _prefetch_owners: dict[ObjectKey, int] = field(default_factory=dict, repr=False)
    extras: dict[str, Any] = field(default_factory=dict)
    # Only one identity context is retained; switching context drops old keys.
    _object_key_identity: tuple[str, int, int | None, str, tuple[int, ...]] | None = (
        field(default=None, repr=False)
    )
    _object_key_groups: list[list[ObjectKey]] = field(default_factory=list, repr=False)
    _object_key_chunks: int = field(default=0, repr=False)
    _lookup_generation: int = field(default=0, repr=False)
    _lookup_observation: LookupObservation | None = field(default=None, repr=False)
    _failed_retrieve_releases: set[tuple[int, int, int, int, int]] = field(
        default_factory=set, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_tokens(self, packed_token_ids: bytes) -> None:
        """Update the token sequence (idempotent, replaces not extends).

        Cached hashes and object identities survive identical updates and
        prefix extensions. Replacement or truncation invalidates both caches.

        Args:
            packed_token_ids: Complete packed token sequence, starting at
                position 0.
        """
        with self._lock:
            if not packed_token_ids.startswith(self.tokens):
                self.chunk_hashes.clear()
                self.last_prefix_hash = None
                self.num_chunks_processed = 0
                self._object_key_identity = None
                self._object_key_groups.clear()
                self._object_key_chunks = 0
            self.tokens = packed_token_ids

    @overload
    def get_hashes(self, start: int, end: int) -> list: ...

    @overload
    def get_hashes(self, start: int) -> list: ...

    def get_hashes(self, start: int, end: int | None = None) -> list:
        """Compute and return chunk hashes for the [start, end) token range.

        Internally computes rolling hashes up to end_chunk, skipping
        already-computed chunks.

        Two calling conventions are supported (declared via ``@overload``)::

            get_hashes(start, end)  # explicit end
            get_hashes(start)       # end = last full-chunk boundary

        Args:
            start: Start token index (must be aligned to chunk_size).
            end: End token index (must be aligned to chunk_size).
                When omitted (``None``), automatically set to the last
                full-chunk boundary of the current token sequence.

        Returns:
            List of hash values for chunks in [start_chunk, end_chunk).

        Raises:
            ValueError: If an explicit ``end`` exceeds the current tokens
                (hashing past them yields valid-looking garbage).
        """
        chunk_size = self.hasher.chunk_size
        assert start % chunk_size == 0, (
            f"start ({start}) must be a multiple of chunk_size ({chunk_size})"
        )
        start_chunk = start // chunk_size

        with self._lock:
            held = num_packed_tokens(self.tokens)
            if end is not None and end > held:
                raise ValueError(
                    f"get_hashes end ({end}) exceeds the session's "
                    f"{held} token(s); the session may have "
                    "been recreated after request cleanup"
                )
            if end is None:
                # No explicit end: use the last full-chunk boundary.
                # Lock must be held here because `self.tokens` may be
                # concurrently replaced by `set_tokens` from another thread.
                end = held - (held % chunk_size)
            assert end % chunk_size == 0, (
                f"end ({end}) must be a multiple of chunk_size ({chunk_size})"
            )
            end_chunk = end // chunk_size
            self._compute_hash(end_chunk)
            return self.chunk_hashes[start_chunk:end_chunk]

    def get_object_keys(
        self,
        key: IPCCacheServerKey,
        start: int,
        end: int | None,
        object_group_ids: list[int],
    ) -> list[list[ObjectKey]]:
        """Resolve cached object identities for a chunk-aligned token range.

        Call set_tokens with the request's current prefix first. Keys retain
        group-major, then chunk-major/rank-minor order. Only immutable object
        identities are cached, never residency, locks or allocation generations.
        A different model, world, worker, salt or group selection replaces the
        cached identity context. Returned lists can be modified by the caller.

        Args:
            key: Model, worker and isolation identity for these objects.
            start: Inclusive, chunk-aligned token index.
            end: Exclusive, chunk-aligned token index within held tokens,
                or None to use the last complete chunk under the session lock.
            object_group_ids: Ordered object groups to resolve.

        Returns:
            One list of object keys per requested group.

        Raises:
            ValueError: If the range is invalid or exceeds the held tokens.
        """
        chunk_size = self.hasher.chunk_size
        identity = (
            key.model_name,
            key.world_size,
            key.worker_id,
            key.cache_salt,
            tuple(object_group_ids),
        )
        with self._lock:
            if end is None:
                held = num_packed_tokens(self.tokens)
                end = held - held % chunk_size
            if (
                start < 0
                or start > end
                or start % chunk_size
                or end % chunk_size
                or end > num_packed_tokens(self.tokens)
            ):
                raise ValueError("Object key range must cover held, complete chunks")
            if identity != self._object_key_identity:
                self._object_key_identity = identity
                self._object_key_groups = [[] for _ in object_group_ids]
                self._object_key_chunks = 0
            end_chunk = end // chunk_size
            self._compute_hash(end_chunk)
            if end_chunk > self._object_key_chunks:
                hashes = [
                    TokenHasher.hash_to_bytes(h)
                    for h in self.chunk_hashes[self._object_key_chunks : end_chunk]
                ]
                additions = ipc_key_to_object_keys(key, hashes, object_group_ids)
                for cached, added in zip(
                    self._object_key_groups, additions, strict=True
                ):
                    cached.extend(added)
                self._object_key_chunks = end_chunk
            ranks = key.world_size if key.worker_id is None else 1
            first = start // chunk_size * ranks
            last = end_chunk * ranks
            return [group[first:last] for group in self._object_key_groups]

    def _compute_hash(self, end_chunk: int) -> None:
        """Compute rolling hashes up to end_chunk.

        Uses cached state to skip already-computed chunks.

        Args:
            end_chunk: Compute hashes up to (but not including) this chunk.
        """
        stride = self.hasher.chunk_size * TOKEN_STRIDE
        # Slicing a memoryview does not copy; slicing the bytes would.
        view = memoryview(self.tokens)

        while self.num_chunks_processed < end_chunk:
            cs = self.num_chunks_processed * stride
            chunk = view[cs : cs + stride]

            prefix = (
                self.last_prefix_hash
                if self.last_prefix_hash is not None
                else self.hasher.none_hash
            )
            h = self.hasher.hash_packed_chunk(chunk, prefix)
            self.last_prefix_hash = h
            self.chunk_hashes.append(h)
            self.num_chunks_processed += 1

    def begin_lookup(
        self,
        key: IPCCacheServerKey,
        group_windows: tuple[int, ...],
    ) -> int:
        """Reset per-lookup state and return the new lookup generation."""
        with self._lock:
            self.lookup_ipc_key = key
            self.prefetch_hit_chunks = -1
            self.prefetch_locked_gids = ()
            self.prefetch_group_windows = group_windows
            self._lookup_generation += 1
            self._failed_retrieve_releases.clear()
            self._prefetch_owners.clear()
            self._lookup_observation = None
            return self._lookup_generation

    def is_lookup_current(self, generation: int) -> bool:
        """Whether this generation still owns the session's lookup state."""
        with self._lock:
            return self._lookup_generation == generation

    def record_prefetch_result(
        self,
        hit_chunks: int,
        locked_gids: tuple[int, ...],
        l1_owners: dict[ObjectKey, int] | None = None,
        observation: LookupObservation | None = None,
        *,
        lookup_generation: int | None = None,
    ) -> bool:
        """Record a result unless its supplied lookup generation was replaced."""
        with self._lock:
            if (
                lookup_generation is not None
                and lookup_generation != self._lookup_generation
            ):
                return False
            self.prefetch_hit_chunks = hit_chunks
            self.prefetch_locked_gids = locked_gids
            self._prefetch_owners = dict(l1_owners) if l1_owners is not None else {}
            self._lookup_observation = observation
            return True

    def get_lookup_observation(self) -> LookupObservation | None:
        """Return the consumed lookup receipt, or None before consumption.

        Reading this snapshot does not extend the session or any read lock.
        """
        with self._lock:
            return self._lookup_observation

    def get_prefetch_owners(self) -> dict[ObjectKey, int]:
        """Snapshot the L1 owners whose read locks this session's lookup retained."""
        with self._lock:
            return self._prefetch_owners.copy()

    def prepare_failed_retrieve_release(
        self,
        key: IPCCacheServerKey,
    ) -> tuple[int, tuple[int, ...], tuple[int, ...], int] | None:
        """Return a stable snapshot for a failed worker's lock release."""
        if key.worker_id is None:
            return None

        with self._lock:
            lookup_key = self.lookup_ipc_key
            hit_chunks = self.prefetch_hit_chunks
            if lookup_key is None or hit_chunks < 0:
                return None

            same_lookup = (
                key.model_name == lookup_key.model_name
                and key.world_size == lookup_key.world_size
                and key.token_bytes == lookup_key.token_bytes
                and key.cache_salt == lookup_key.cache_salt
                and key.start >= lookup_key.start
                and key.end <= lookup_key.end
            )
            if not same_lookup:
                return None

            return (
                hit_chunks,
                self.prefetch_locked_gids,
                self.prefetch_group_windows,
                self._lookup_generation,
            )

    def claim_failed_retrieve_release(
        self,
        instance_id: int,
        key: IPCCacheServerKey,
        lookup_generation: int,
    ) -> bool:
        """Atomically claim one failed worker's prepared lock release.

        A scheduler lookup acquires read locks for every KV worker (or one
        reader count per TP worker for MLA), while RETRIEVE responses are
        per worker instance.  When a worker has lost its GPU registration,
        only that instance's share may be released.  Claiming after key
        resolution makes duplicate failed RETRIEVEs idempotent without
        consuming the claim when resolution itself fails.

        ``False`` is also returned when the session cannot prove ownership of
        the requested range.  In that case it is safer to leave the lock to
        its TTL than to decrement a concurrent request's anonymous L1 count.
        """
        if key.worker_id is None:
            return False

        with self._lock:
            lookup_key = self.lookup_ipc_key
            if lookup_key is None or lookup_generation != self._lookup_generation:
                return False

            same_lookup = (
                key.model_name == lookup_key.model_name
                and key.world_size == lookup_key.world_size
                and key.token_bytes == lookup_key.token_bytes
                and key.cache_salt == lookup_key.cache_salt
                and key.start >= lookup_key.start
                and key.end <= lookup_key.end
            )
            if not same_lookup:
                return False

            owner = (
                lookup_generation,
                instance_id,
                key.worker_id,
                key.start,
                key.end,
            )
            if owner in self._failed_retrieve_releases:
                return False
            self._failed_retrieve_releases.add(owner)
            return True


class SessionManager:
    """Thread-safe manager for per-request sessions."""

    DEFAULT_SESSION_TTL = 600  # 10 minutes
    DEFAULT_CLEANUP_INTERVAL = 60.0

    def __init__(
        self,
        hasher: TokenHasher,
        ttl: float = DEFAULT_SESSION_TTL,
        cleanup_interval: float | None = DEFAULT_CLEANUP_INTERVAL,
    ) -> None:
        self._hasher = hasher
        self._ttl = ttl
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        #: Callbacks invoked for every destroyed session, so owners of
        #: per-request state (e.g. entries in ``Session.extras``) can release
        #: it however the request ended. See :meth:`_notify_destroyed`.
        self.destroy_listeners: list[Callable[[Session], None]] = []
        self._cleanup_interval = cleanup_interval
        self._cleanup_thread: PeriodicThread | None = None
        if cleanup_interval is not None and cleanup_interval > 0:
            self._cleanup_thread = self._create_cleanup_thread(cleanup_interval)
            self._cleanup_thread.start()

    def get_or_create(self, request_id: str) -> Session:
        """Get existing session or create a new one.

        Args:
            request_id: Unique request identifier.

        Returns:
            The Session for this request_id.
        """
        with self._lock:
            if request_id not in self._sessions:
                self._sessions[request_id] = Session(
                    request_id=request_id, hasher=self._hasher
                )
                logger.debug("Created session for request_id=%s", request_id)
            return self._sessions[request_id]

    def get(self, request_id: str) -> Optional[Session]:
        """Return an existing session without creating ownership state."""
        with self._lock:
            return self._sessions.get(request_id)

    def remove(self, request_id: str) -> Optional[Session]:
        """Remove a session by request_id.

        Args:
            request_id: Unique request identifier.

        Returns:
            The removed session, or None if no session was found.
        """
        with self._lock:
            if request_id not in self._sessions:
                return None
            session = self._sessions[request_id]
            del self._sessions[request_id]
            logger.debug("Removed session for request_id=%s", request_id)
        self._notify_destroyed(session)
        return session

    def cleanup_expired(self) -> int:
        """Remove sessions that have exceeded their TTL.

        Returns:
            Number of sessions removed.
        """
        now = time.time()
        expired: list[Session] = []
        with self._lock:
            for session in self._sessions.values():
                if now - session.created_at > self._ttl:
                    expired.append(session)
            for session in expired:
                del self._sessions[session.request_id]

        for session in expired:
            self._notify_destroyed(session)
        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))
        return len(expired)

    def _notify_destroyed(self, session: Session) -> None:
        """Run every destroy listener for one removed session.

        Called without the manager lock held: a listener may take other locks
        (e.g. the storage manager's). A listener that raises is logged and
        skipped so destruction always completes.

        Args:
            session: The session that was just removed.
        """
        for listener in list(self.destroy_listeners):
            try:
                listener(session)
            except Exception:
                logger.exception(
                    "Session destroy listener failed for request_id=%s",
                    session.request_id,
                )

    def active_count(self) -> int:
        """Return the number of active sessions.

        Returns:
            Number of currently tracked sessions.
        """
        with self._lock:
            return len(self._sessions)

    def close(self) -> None:
        """Stop the background cleanup thread and unregister it."""
        if self._cleanup_thread is None:
            return

        PeriodicThreadRegistry.get_instance().unregister(self._cleanup_thread.name)
        self._cleanup_thread.stop()
        self._cleanup_thread = None

    def _create_cleanup_thread(self, cleanup_interval: float) -> PeriodicThread:
        def execute_cleanup() -> ThreadRunSummary:
            removed = self.cleanup_expired()
            return ThreadRunSummary(
                success=True,
                message=f"Removed {removed} expired sessions",
                extra_info={"removed_sessions": str(removed)},
            )

        return create_periodic_thread(
            name=f"SessionManager-cleanup-thread-{id(self):x}",
            interval=cleanup_interval,
            execute_fn=execute_cleanup,
            level=ThreadLevel.MEDIUM,
        )
