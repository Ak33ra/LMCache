Bounded L1 admission
====================

The distributed ``L1Manager`` can limit retained and temporary allocations
independently inside one CPU pool. This internal API is an admission primitive
for partitioning experiments. It does not enable partitioning in the MP server.

Pass optional limits when constructing a manager:

.. code-block:: python

   from lmcache.v1.distributed.internal_api import (
       L1AdmissionLimits,
       L1WriteRetention,
   )
   from lmcache.v1.distributed.l1_manager import L1Manager

   manager = L1Manager(
       config,
       memory_manager=pool,
       admission_limits=L1AdmissionLimits(
           retained_bytes=8192,
           temporary_bytes=4096,
       ),
   )
   results = manager.reserve_write_operations(
       keys,
       [L1WriteRetention.PREFER_RETAIN] * len(keys),
       layout,
       tag="example",
   )

``config`` and the unused segmented CPU ``pool`` must describe the same backing
capacity and alignment. ``keys`` must be distinct and ``layout`` must fit the
pool. ``RETAIN`` requires retained quota, ``TEMPORARY`` requires temporary quota,
and ``PREFER_RETAIN`` tries retained quota before bounded temporary fallback.
Results report each key's admission independently. Quotas charge aligned
physical allocation sizes, including unpublished staging and pinned discards.

Complete every successful write handle through ``finish_write_operations``
after local buffer access ends. Pass failed handles in ``discard``. Returned
memory objects are borrowed; do not free or unpin them directly. Use exact read
handles for consumers whose buffer access can outlive serving reservation TTLs.

``get_admission_usage()`` returns class usage and limits.
``set_admission_limits()`` updates quotas without moving or freeing live data.
First activation on an existing manager requires it to be empty.
``set_admission_state(L1AdmissionState.QUIESCING)`` refuses new readers and
writers while allowing held operations to complete. Reopen with ``OPEN``.
Neither quiescing nor TTL expiry proves that native buffer access has ended.

The existing L1 eviction controller uses retained quota pressure and sweeps
abandoned temporary residents after their readers expire. Without limits,
existing allocation behavior is preserved and ``PREFER_RETAIN`` means
``RETAIN``. The legacy boolean ``reserve_write`` API requires unbounded mode.
