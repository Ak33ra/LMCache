Mooncake Store
==============

An L2 adapter backed by the native C++ Mooncake Store connector.  Uses
`Mooncake <https://github.com/kvcache-ai/Mooncake>`_ for high-performance
distributed KV cache storage with RDMA support.

When Mooncake is configured with ``"protocol": "rdma"``, LMCache must also
have a valid contiguous L1 memory region available.  The distributed storage
manager passes this L1 memory descriptor to the adapter factory automatically
in MP mode.  If the descriptor is missing or invalid, adapter creation fails
with ``ValueError`` instead of silently falling back to a non-RDMA path.

**Prerequisites -- Building with Mooncake support:**

The Mooncake extension is **not** built by default.  You must explicitly
enable it:

.. code-block:: bash

    BUILD_MOONCAKE=1 pip install -e . --verbose

The ``BUILD_MOONCAKE`` environment variable controls compilation:

- ``BUILD_MOONCAKE=1``: Enable the Mooncake C++ extension.
- ``BUILD_MOONCAKE=0``: Force disable (highest priority), even if
  ``MOONCAKE_INCLUDE_DIR`` is set.
- **Not set**: Falls back to checking ``MOONCAKE_INCLUDE_DIR`` for
  backward compatibility.  If ``MOONCAKE_INCLUDE_DIR`` is also unset,
  the extension is skipped.

If the Mooncake headers are not installed in the system include path
(e.g., ``/usr/local/include``), you must point to them explicitly:

.. code-block:: bash

    BUILD_MOONCAKE=1 \
    MOONCAKE_INCLUDE_DIR=/path/to/mooncake/include \
    MOONCAKE_LIB_DIR=/path/to/mooncake/lib \
    pip install -e . --verbose

**LMCache-specific fields:**

- ``num_workers``: Number of C++ worker threads for the shared pool
  (default ``4``, must be > 0).

- ``per_op_workers`` (``dict[str, int]``, optional): A dict mapping lane keys
  to dedicated worker thread counts.  Supported keys:

  - ``"lookup"`` — threads for ``EXISTS`` operations.
  - ``"retrieve"`` — threads for ``GET`` / load operations.
  - ``"store"`` — threads for ``SET`` / put operations.
  - ``"delete"`` — threads for ``DELETE`` operations.

  Operations whose lane key is **not** present in the dict use the
  shared ``num_workers`` pool.  There is no requirement to set all
  keys — you can configure only the lanes that need dedicated pools.

**Mooncake fields:**

All other keys in the JSON config (except ``type``, ``num_workers``,
``per_op_workers``, and ``eviction``) are forwarded **as-is** to Mooncake's
``setup_internal(ConfigDict)``.  Refer to the
`Mooncake documentation <https://github.com/kvcache-ai/Mooncake>`_
for available setup keys (e.g., ``local_hostname``,
``metadata_server``, ``master_server_addr``, ``protocol``,
``rdma_devices``, ``global_segment_size``).

**Configuration example:**

.. code-block:: bash

    # Shared pool (default)
    --l2-adapter '{
      "type": "mooncake_store",
      "num_workers": 4,
      "local_hostname": "node01",
      "metadata_server": "http://localhost:8080/metadata",
      "master_server_addr": "localhost:50051",
      "protocol": "tcp",
      "local_buffer_size": "3221225472",
      "global_segment_size": "3221225472"
    }'

    # Per-operation pools (GET-heavy workload)
    --l2-adapter '{
      "type": "mooncake_store",
      "per_op_workers": {
        "lookup": 2,
        "retrieve": 16,
        "store": 4
      },
      "local_hostname": "node01",
      "metadata_server": "http://localhost:8080/metadata",
      "master_server_addr": "localhost:50051",
      "protocol": "tcp"
    }'

For full Mooncake setup instructions (master service, metadata server,
etc.), see `Mooncake <https://github.com/kvcache-ai/Mooncake>`_ .

**RDMA notes:**

- ``protocol: "rdma"`` requires a valid LMCache L1 memory descriptor.
- When using ``protocol: "rdma"``, it is recommended to disable lazy L1
  allocation with ``--no-l1-use-lazy`` so the L1 buffer is fully allocated
  before Mooncake registers it.
- ``protocol: "tcp"`` does not require L1 preregistration.
- If Mooncake RDMA initialization fails at adapter creation time, verify that
  LMCache L1 memory is enabled and that the descriptor has a non-zero pointer
  and size.

Experimental CPU partition controls
-----------------------------------

The native connector accepts ``partition_controls=True`` only with the fenced
CPU TCP SDK fork. This mode requires zero setup shared capacity and dynamic
local-buffer registration. It exposes explicit local registration, removable
shared segment allocation/release, accounting snapshots, and retryable close.
It does not enable partitioning in the MP server by itself.

Control calls return typed results. ``BUSY`` retains all ownership for retry.
Failed shared allocation can retain physical backing, reported separately from
mounted shared segments. Private deregistration releases transport registration
only; the caller still owns its backing. Keep private backing alive until
explicit cleanup succeeds.

Keep submitted buffers alive until their terminal results are returned by
``drain_completions()``. Unregister and close return ``BUSY`` while accepted
work is queued, active, or completed but undrained. Drain the results and retry.
Partition get/set reject unregistered ranges at admission. All four submission
methods raise ``RuntimeError`` once close starts or succeeds. An exception
during native submission can leave unresolved ownership without a future ID;
in that case retain the connector and backing until process exit.

The build and live-master verification commands are documented in
``docs/design/v1/distributed/l2_adapters/mooncake_partition_controls.md``.
The checked-in probe is
``csrc/storage_backends/mooncake/verify_partition_controls.py``.

Opt-in private/shared partitioning
----------------------------------

The MP server accepts ``--kv-partition`` with four explicit fields:

.. code-block:: json

    {"total_bytes": 92274688, "staging_bytes": 16777216,
     "sdk_scratch_bytes": 8388608, "alpha": 0.5}

This reserves 16 MiB for temporary L1 operations and 8 MiB for SDK scratch.
The remaining 64 MiB is split into private and shared backing in whole native
segments. With the tested 16 MiB granule, alpha 0.5 gives 32 MiB to each.
The total bounds managed cache backing. RPC buffers, worker tensors, runtime
metadata, and process RSS are separate.

This experimental path requires the fenced CPU TCP SDK, the default MP engine,
ZMQ, engine-driven typed-inline transfer, and an empty SHM name. Set
``MC_TCP_REQUIRE_GENERATION=1`` and ``MC_NOF_BACKEND=none``. Configure exactly
one Mooncake adapter with ``protocol=tcp``, ``global_segment_size=0``, and
``local_buffer_size`` equal to the decimal SDK scratch byte count. Partitioning
owns L1 capacity, backing, and policy selection. Conflicting legacy settings,
GPU/SHM transfer, P2P, plugins, run-script access, and trace capture are rejected.

While shared capacity is nonzero, new writes become visible only after shared
publication. Reads can retain an independent private copy, or use temporary
storage when private quota is full. At alpha zero, reads use temporary storage
and leave no private copy. At alpha one, writes stay private and remote lookup
is disabled. The default mode without ``--kv-partition`` remains available.

Each accepted ZMQ request has an 8 MiB wire-payload ceiling, with at most four
accepted requests and a 32 MiB per-request inline-copy workspace ceiling.
Oversized requests fail before copying into cache backing. These limits do not
bound rejected input transients, socket buffers, or whole-process RSS.

The HTTP management port exposes cached ownership and asynchronous resize:

.. code-block:: bash

    curl -fsS http://127.0.0.1:8000/partition | python -m json.tool
    curl -fsS -H 'Content-Type: application/json' \
      -d '{"request_id":"manual-1","alpha":1.0}' \
      http://127.0.0.1:8000/partition/resize | python -m json.tool
    curl -fsS http://127.0.0.1:8000/partition/resize/manual-1 \
      | python -m json.tool

Admission pauses while resize drains operations and reclaims backing. A resize
can evict cached objects. Only confirmed release permits opposite-owner growth.
Repeated request IDs with the same alpha return the same job; a changed target
or another concurrent resize returns HTTP 409. Failed jobs remain quiesced with
their actual backing charged. Inspect status before retrying the same request.

``USER_VERIFY.md`` records the focused CPU checks. The surrounding partitioning
project supplies the pinned SDK build recipe and live two-server harness.
