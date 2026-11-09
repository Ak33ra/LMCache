# Session-scoped lookup observations

`GET /cache/lookups/{request_id}` exposes the most recently consumed lookup
result from the standard `LookupModule` while its request session exists.
Blend lookups do not publish this receipt. It does not initiate lookup, refresh
locks, inspect current cache contents or retain an independent history. Ending
or expiring the session removes the observation. Beginning another lookup
clears it; a pending lookup returns HTTP 409 and an absent session returns 404.

The immutable receipt records the absolute hit-prefix length and per-grid-row
counts of keys already in L1 versus keys loaded from L2. Counts come from the
final `PrefetchResult` bitmaps, not a separate prefix fold over each source.
An alternating L1/L2 prefix therefore retains its actual source counts.
Partition quiescence declines new prefetch admission with zero-hit bitmaps of
the requested row layout. Its receipt contains zero source counts per row,
preserves any engine-covered prefix offset, and holds no cache read locks.

L1 is an interface origin, not a retention policy. A hit may reuse a temporary
buffer. L2 does not identify a physical local or remote replica. Consumers
must name those fields accordingly. A successful lookup is not proof that a
later retrieve succeeded; the caller must verify the retrieved bytes.

This read-only observation leaves existing integer lookup RPCs and lock
ownership unchanged. The session lock protects publication and reads.
Each prefetch job captures its original session and lookup generation. Late
completions cannot recreate an ended session or overwrite a replacement lookup;
their status is zero. Rejected stale results leave read locks to the existing
controller and TTL handling instead of releasing anonymous locks speculatively.
