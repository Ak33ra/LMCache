// SPDX-License-Identifier: Apache-2.0
#pragma once

// Standard
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Third Party
#include "real_client.h"

// First Party
#include "../connector_base.h"
#include "partition_controls.h"

namespace lmcache {
namespace connector {

// ConfigDict mirrors mooncake::ConfigDict
// (std::unordered_map<std::string, std::string>).
using ConfigDict = std::unordered_map<std::string, std::string>;

// Per-worker connection state for the Mooncake connector.
// Each worker holds a raw pointer to the shared
// RealClient (owned by MooncakeConnector).
struct WorkerMooncakeConn {
  mooncake::RealClient* client{nullptr};
};

struct L1RegistrationConfig {
  bool enabled{false};
  std::uintptr_t base{0};
  size_t size{0};

  bool is_valid() const { return enabled && base != 0 && size != 0; }
};

class MooncakeConnector : public ConnectorBase<WorkerMooncakeConn> {
 public:
  MooncakeConnector(ConfigDict config, int num_workers,
                    L1RegistrationConfig l1_registration = {},
                    WorkerPoolConfig worker_pool_config = {},
                    bool partition_controls = false);
  ~MooncakeConnector() override;

  /** Submit a batch; partition mode retains registrations until drain.
   * Callers retain backing until terminal completion. Closing rejects before
   * admission, and partition get/set require registered ranges at admission.
   */
  uint64_t submit_batch_get(const std::vector<std::string>& keys,
                            const std::vector<void*>& bufs,
                            const std::vector<size_t>& lens,
                            size_t batch_chunk_num_bytes) override;
  /** Submit SET with the same partition admission and retention as GET. */
  uint64_t submit_batch_set(const std::vector<std::string>& keys,
                            const std::vector<void*>& bufs,
                            const std::vector<size_t>& lens,
                            size_t batch_chunk_num_bytes) override;
  /** Submit EXISTS, retaining partition ownership until completion drain. */
  uint64_t submit_batch_exists(const std::vector<std::string>& keys) override;
  /** Submit DELETE, retaining partition ownership until completion drain. */
  uint64_t submit_batch_delete(const std::vector<std::string>& keys) override;

  /** Drain terminal results and release their partition admission ownership. */
  std::vector<Completion> drain_completions() override;

  /** Return the last confirmed snapshot without waiting for SDK access. */
  PartitionStatus partition_status() const;

  /** Register caller-owned CPU backing; never mount it as shared capacity. */
  PartitionControlResult register_l1_buffer(std::uintptr_t base, size_t size);

  /** Deregister an exact ID, or return BUSY until accepted work is drained.
   * Successful deregistration does not free caller-owned backing.
   */
  PartitionControlResult unregister_l1_buffer(uint64_t id);

  /** Allocate removable SDK-owned shared backing, retaining failed charges. */
  PartitionControlResult allocate_shared_segment(size_t size);

  /** Synchronously unmount, fence and free one shared segment, or retain it. */
  PartitionControlResult release_shared_segment(const std::string& id);

  /** Close after accepted futures drain, or return BUSY with ownership held. */
  PartitionControlResult close_partition();

  /** Serialize worker shutdown with explicit partition teardown. */
  void close() override;

 protected:
  WorkerMooncakeConn create_connection() override;

  void do_single_get(WorkerMooncakeConn& conn, const std::string& key,
                     void* buf, size_t len, size_t chunk_size) override;

  void do_single_set(WorkerMooncakeConn& conn, const std::string& key,
                     const void* buf, size_t len, size_t chunk_size) override;

  bool do_single_exists(WorkerMooncakeConn& conn,
                        const std::string& key) override;

  bool do_single_delete(WorkerMooncakeConn& conn,
                        const std::string& key) override;

  void on_workers_stopped() override;

  size_t choose_num_tiles(Op op, size_t num_items) const override;

  void do_batch_get(WorkerMooncakeConn& conn, const Request& req) override;

  void do_batch_set(WorkerMooncakeConn& conn, const Request& req) override;

  void do_batch_exists(WorkerMooncakeConn& conn, const Request& req) override;

  void do_batch_delete(WorkerMooncakeConn& conn, const Request& req) override;

 private:
  uint64_t submit_batch(Op op, const std::vector<std::string>& keys,
                        const std::vector<void*>& bufs,
                        const std::vector<size_t>& lens,
                        size_t batch_chunk_num_bytes);
  void ensure_registered(const void* buf, size_t len);
  void preregister_l1_memory(std::uintptr_t base, size_t size);
  void unregister_all_buffers() noexcept;
  void validate_partition_config() const;
  void refresh_partition_status();
  PartitionControlResult control_result(PartitionControlCode code,
                                        int sdk_error = 0) const;
  PartitionControlCode control_admission() const;

  bool partition_controls_{false};
  bool partition_closing_{false};
  uint64_t next_registration_id_{1};
  std::unordered_map<uint64_t, LocalBufferRegistration> local_buffers_;
  mutable std::shared_mutex sdk_mutex_;
  mutable std::mutex status_mutex_;
  std::mutex close_mutex_;
  // Admission and drain serialize so a fast completion cannot precede its ID.
  // An empty ID retains an ambiguous submission exception until process exit.
  std::mutex submission_mutex_;
  std::list<std::optional<uint64_t>> pending_submissions_;
  PartitionStatus partition_status_;

  // Shared Mooncake RealClient instance.
  std::shared_ptr<mooncake::RealClient> client_;

  // The original config dict (kept for diagnostics).
  ConfigDict config_;
  L1RegistrationConfig l1_registration_;
  size_t preregistered_block_size_{0};
};

}  // namespace connector
}  // namespace lmcache
