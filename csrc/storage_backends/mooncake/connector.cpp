// SPDX-License-Identifier: Apache-2.0

// Standard
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

// Third Party
#include "config.h"
#include "cachelib_memory_allocator/Slab.h"

// Local
#include "connector.h"

namespace lmcache {
namespace connector {

namespace {

template <typename T>
void ensure_batch_result_size(const std::vector<T>& results, size_t expected,
                              const char* op_name) {
  if (results.size() != expected) {
    throw std::runtime_error(std::string("Mooncake ") + op_name + " returned " +
                             std::to_string(results.size()) + " results for " +
                             std::to_string(expected) + " keys");
  }
}

template <typename Operation>
int control_sdk_call(Operation&& operation) {
  try {
    return operation();
  } catch (const std::exception& error) {
    fprintf(stderr, "[LMCache Mooncake] partition SDK exception: %s\n",
            error.what());
    return -1;
  }
}

}  // namespace

MooncakeConnector::MooncakeConnector(ConfigDict config, int num_workers,
                                     L1RegistrationConfig l1_registration,
                                     WorkerPoolConfig worker_pool_config,
                                     bool partition_controls)
    : ConnectorBase(num_workers, std::move(worker_pool_config)),
      config_(std::move(config)),
      l1_registration_(l1_registration),
      partition_controls_(partition_controls) {
  if (partition_controls_) validate_partition_config();
  // Create a RealClient via the static factory.
  client_ = mooncake::RealClient::create();
  if (!client_) {
    throw std::runtime_error("Failed to create mooncake RealClient");
  }

  // Forward the config dict to setup_internal().
  mooncake::ConfigDict mc_config(config_.begin(), config_.end());
  auto result = client_->setup_internal(mc_config);
  if (!result.has_value()) {
    throw std::runtime_error("Mooncake setup_internal failed");
  }

  if (l1_registration.is_valid()) {
    preregister_l1_memory(l1_registration.base, l1_registration.size);
  }

  if (partition_controls_) {
    refresh_partition_status();
    if (!partition_status().enabled) {
      throw std::runtime_error(
          "Mooncake strict CPU TCP ownership is unavailable");
    }
  }
  start_workers();  // IMPORTANT: call at END of ctor
}

MooncakeConnector::~MooncakeConnector() {
  if (partition_controls_) {
    const auto result = close_partition();
    if (result.code != PartitionControlCode::SUCCESS) {
      fprintf(
          stderr,
          "[LMCache Mooncake] unresolved partition ownership at destruction; "
          "caller must retain private backing after close failure\n");
      ConnectorBase<WorkerMooncakeConn>::close();
    }
  } else {
    close();
    if (client_) client_->tearDownAll();
  }
}

void MooncakeConnector::close() {
  if (partition_controls_) {
    const auto result = close_partition();
    if (result.code != PartitionControlCode::SUCCESS) {
      throw std::runtime_error(
          "Mooncake partition close incomplete; retain backing and retry "
          "close_partition()");
    }
    return;
  }
  std::lock_guard<std::mutex> guard(close_mutex_);
  ConnectorBase<WorkerMooncakeConn>::close();
}

WorkerMooncakeConn MooncakeConnector::create_connection() {
  WorkerMooncakeConn conn;
  conn.client = client_.get();
  return conn;
}

void MooncakeConnector::do_single_get(WorkerMooncakeConn& conn,
                                      const std::string& key, void* buf,
                                      size_t len, size_t chunk_size) {
  std::shared_lock<std::shared_mutex> access(sdk_mutex_);
  if (partition_closing_)
    throw std::runtime_error("Mooncake partition is closing");
  (void)chunk_size;
  ensure_registered(buf, len);
  int64_t bytes_read = conn.client->get_into(key, buf, len);
  if (bytes_read <= 0) {
    throw std::runtime_error("Mooncake get_into failed for key: " + key);
  }
}

void MooncakeConnector::do_single_set(WorkerMooncakeConn& conn,
                                      const std::string& key, const void* buf,
                                      size_t len, size_t chunk_size) {
  std::shared_lock<std::shared_mutex> access(sdk_mutex_);
  if (partition_closing_)
    throw std::runtime_error("Mooncake partition is closing");
  (void)chunk_size;
  ensure_registered(buf, len);
  int rc = conn.client->put_from(key, const_cast<void*>(buf), len);
  if (rc != 0) {
    throw std::runtime_error("Mooncake put_from failed for key: " + key);
  }
}

bool MooncakeConnector::do_single_exists(WorkerMooncakeConn& conn,
                                         const std::string& key) {
  std::shared_lock<std::shared_mutex> access(sdk_mutex_);
  if (partition_closing_)
    throw std::runtime_error("Mooncake partition is closing");
  // isExist returns: 1=exists, 0=not, -1=error
  int result = conn.client->isExist(key);
  if (result < 0) {
    fprintf(stderr,
            "[LMCache EXISTS] key %s failed: Mooncake isExist returned %d\n",
            key.c_str(), result);
    return false;
  }
  return result == 1;
}

bool MooncakeConnector::do_single_delete(WorkerMooncakeConn& conn,
                                         const std::string& key) {
  std::shared_lock<std::shared_mutex> access(sdk_mutex_);
  if (partition_closing_)
    throw std::runtime_error("Mooncake partition is closing");
  int rc = conn.client->remove(key, /*force=*/true);
  if (rc != 0) {
    fprintf(stderr,
            "[LMCache DELETE] key %s failed: Mooncake remove returned %d\n",
            key.c_str(), rc);
    return false;
  }
  return true;
}

void MooncakeConnector::on_workers_stopped() {
  if (!partition_controls_) unregister_all_buffers();
}

size_t MooncakeConnector::choose_num_tiles(Op op, size_t num_items) const {
  (void)op;
  (void)num_items;
  // Mooncake's batch APIs already parallelize internally. Keep each LMCache
  // batch as one tile so we do not split a single backend batch across workers.
  return 1;
}

void MooncakeConnector::do_batch_get(WorkerMooncakeConn& conn,
                                     const Request& req) {
  std::shared_lock<std::shared_mutex> access(sdk_mutex_);
  if (partition_closing_)
    throw std::runtime_error("Mooncake partition is closing");
  // Keep GET tolerant at per-key granularity: registration failures should
  // only zero that key. Keep the no-failure path on Mooncake's batch API, and
  // fall back to ConnectorBase's singleton-style handling on error.
  for (size_t i = 0; i < req.buf_ptrs.size(); ++i) {
    try {
      ensure_registered(req.buf_ptrs[i], req.buf_lens[i]);
    } catch (const std::exception& e) {
      fprintf(stderr,
              "[LMCache GET] Mooncake buffer registration failed for key %zu, "
              "falling back to per-key get: %s\n",
              i, e.what());
      access.unlock();
      ConnectorBase<WorkerMooncakeConn>::do_batch_get(conn, req);
      return;
    }
  }

  auto results =
      conn.client->batch_get_into(req.keys, req.buf_ptrs, req.buf_lens);
  ensure_batch_result_size(results, req.keys.size(), "batch_get_into");

  for (size_t i = 0; i < results.size(); ++i) {
    if (results[i] <= 0) {
      req.batch->per_key_results[req.start_idx + i] = 0;
      fprintf(stderr,
              "[LMCache GET] key %s failed: Mooncake batch_get_into "
              "returned %lld\n",
              req.keys[i].c_str(), static_cast<long long>(results[i]));
      continue;
    }
    req.batch->per_key_results[req.start_idx + i] = 1;
  }
}

void MooncakeConnector::do_batch_set(WorkerMooncakeConn& conn,
                                     const Request& req) {
  std::shared_lock<std::shared_mutex> access(sdk_mutex_);
  if (partition_closing_)
    throw std::runtime_error("Mooncake partition is closing");
  for (size_t i = 0; i < req.buf_ptrs.size(); ++i) {
    ensure_registered(req.buf_ptrs[i], req.buf_lens[i]);
  }

  auto results =
      conn.client->batch_put_from(req.keys, req.buf_ptrs, req.buf_lens);
  ensure_batch_result_size(results, req.keys.size(), "batch_put_from");

  for (size_t i = 0; i < results.size(); ++i) {
    if (results[i] != 0) {
      throw std::runtime_error("Mooncake batch_put_from failed for key: " +
                               req.keys[i]);
    }
  }
}

void MooncakeConnector::do_batch_exists(WorkerMooncakeConn& conn,
                                        const Request& req) {
  std::shared_lock<std::shared_mutex> access(sdk_mutex_);
  if (partition_closing_)
    throw std::runtime_error("Mooncake partition is closing");
  auto results = conn.client->batchIsExist(req.keys);
  ensure_batch_result_size(results, req.keys.size(), "batchIsExist");

  for (size_t i = 0; i < results.size(); ++i) {
    if (results[i] < 0) {
      fprintf(stderr,
              "[LMCache EXISTS] key %s failed: Mooncake batchIsExist "
              "returned %d\n",
              req.keys[i].c_str(), results[i]);
      req.batch->per_key_results[req.start_idx + i] = 0;
      continue;
    }
    req.batch->per_key_results[req.start_idx + i] = results[i] == 1 ? 1 : 0;
  }
}

void MooncakeConnector::do_batch_delete(WorkerMooncakeConn& conn,
                                        const Request& req) {
  std::shared_lock<std::shared_mutex> access(sdk_mutex_);
  if (partition_closing_)
    throw std::runtime_error("Mooncake partition is closing");
  auto results = conn.client->batchRemove(req.keys, /*force=*/true);
  ensure_batch_result_size(results, req.keys.size(), "batchRemove");

  for (size_t i = 0; i < results.size(); ++i) {
    if (results[i] != 0) {
      fprintf(stderr,
              "[LMCache DELETE] key %s failed: Mooncake batchRemove "
              "returned %d\n",
              req.keys[i].c_str(), results[i]);
      req.batch->per_key_results[req.start_idx + i] = 0;
      continue;
    }
    req.batch->per_key_results[req.start_idx + i] = 1;
  }
}

void MooncakeConnector::ensure_registered(const void* buf, size_t len) {
  if (partition_controls_) {
    const auto address = reinterpret_cast<std::uintptr_t>(buf);
    if (!address || !len ||
        len > std::numeric_limits<std::uintptr_t>::max() - address) {
      throw std::runtime_error("Invalid Mooncake local buffer range");
    }
    for (const auto& [id, region] : local_buffers_) {
      if (region.state == LocalBufferState::REGISTERED &&
          address >= region.base && address - region.base < region.size &&
          len <= region.size - (address - region.base))
        return;
    }
    throw std::runtime_error(
        "Mooncake buffer has no active local registration");
  }
  if (!l1_registration_.is_valid()) {
    return;
  }
  if (buf == nullptr) {
    throw std::runtime_error(
        "Mooncake buffer registration failed: null buffer");
  }
  if (len == 0) {
    throw std::runtime_error(
        "Mooncake buffer registration failed: zero length");
  }

  if (preregistered_block_size_ == 0) {
    throw std::runtime_error("Mooncake preregistered block size is invalid");
  }

  const auto registered_begin = l1_registration_.base;
  const auto registered_end = registered_begin + l1_registration_.size;
  const auto buf_begin = reinterpret_cast<std::uintptr_t>(buf);
  const auto buf_end = buf_begin + len;

  if (buf_begin < registered_begin || buf_end > registered_end) {
    throw std::runtime_error(
        "Buffer is outside of preregistered L1 region; Mooncake lazy "
        "registration is disabled");
  }

  const auto relative_begin = buf_begin - registered_begin;
  const auto relative_end = (buf_end - 1) - registered_begin;
  const size_t left_region = relative_begin / preregistered_block_size_;
  const size_t right_region = relative_end / preregistered_block_size_;

  if (left_region != right_region) {
    throw std::runtime_error("Buffer crosses preregistered Mooncake regions");
  }
}

void MooncakeConnector::preregister_l1_memory(std::uintptr_t base,
                                              size_t size) {
  if (base == 0 || size == 0) {
    return;
  }

  preregistered_block_size_ = mooncake::globalConfig().max_mr_size == 0
                                  ? size
                                  : mooncake::globalConfig().max_mr_size;
  const size_t max_registration_size = preregistered_block_size_;
  size_t remaining = size;
  auto current = base;

  while (remaining > 0) {
    const size_t segment_size = std::min(remaining, max_registration_size);
    void* segment_ptr = reinterpret_cast<void*>(current);
    const int register_rc = client_->register_buffer(segment_ptr, segment_size);
    if (register_rc != 0) {
      auto rollback_remaining = current - base;
      while (rollback_remaining > 0) {
        const auto rollback_size =
            std::min(rollback_remaining, max_registration_size);
        rollback_remaining -= rollback_size;
        void* rollback_ptr = reinterpret_cast<void*>(base + rollback_remaining);
        try {
          client_->unregister_buffer(rollback_ptr);
        } catch (...) {
          // Keep rolling back the remaining segments so a cleanup failure does
          // not mask the original registration error or strand later regions.
        }
      }
      preregistered_block_size_ = 0;
      throw std::runtime_error("Mooncake preregister_l1_memory failed");
    }

    current += segment_size;
    remaining -= segment_size;
  }
}

void MooncakeConnector::unregister_all_buffers() noexcept {
  if (client_ == nullptr || !l1_registration_.is_valid() ||
      preregistered_block_size_ == 0) {
    return;
  }

  auto current = l1_registration_.base;
  size_t remaining = l1_registration_.size;
  while (remaining > 0) {
    const auto segment_size = std::min(remaining, preregistered_block_size_);
    void* segment_ptr = reinterpret_cast<void*>(current);
    try {
      client_->unregister_buffer(segment_ptr);
    } catch (...) {
      // Preserve noexcept during teardown and keep attempting the remaining
      // segments so earlier failures do not strand later registrations.
    }
    current += segment_size;
    remaining -= segment_size;
  }

  preregistered_block_size_ = 0;
}

PartitionStatus MooncakeConnector::partition_status() const {
  std::lock_guard<std::mutex> guard(status_mutex_);
  return partition_status_;
}

PartitionControlResult MooncakeConnector::control_result(
    PartitionControlCode code, int sdk_error) const {
  PartitionControlResult result;
  result.code = code;
  result.sdk_error = sdk_error;
  result.snapshot = partition_status();
  return result;
}

PartitionControlCode MooncakeConnector::control_admission() const {
  if (!partition_controls_) return PartitionControlCode::UNSUPPORTED;
  if (partition_closing_) return PartitionControlCode::CLOSED;
  return PartitionControlCode::SUCCESS;
}

void MooncakeConnector::validate_partition_config() const {
#ifndef MOONCAKE_FENCED_TCP_OWNERSHIP_API
  throw std::runtime_error("Partition controls require the fenced CPU TCP SDK");
#else
  auto configured = [&](const char* key, const char* value) {
    auto found = config_.find(key);
    return found != config_.end() && found->second == value;
  };
  const char* strict = std::getenv("MC_TCP_REQUIRE_GENERATION");
  const char* nof = std::getenv("MC_NOF_BACKEND");
  const char* hot_cache = std::getenv("MC_STORE_LOCAL_HOT_CACHE_SIZE");
  if (!configured("protocol", "tcp") ||
      !configured("global_segment_size", "0") || !strict ||
      std::string(strict) != "1" || !nof || std::string(nof) != "none" ||
      (hot_cache && std::string(hot_cache) != "0") ||
      l1_registration_.enabled) {
    throw std::invalid_argument(
        "Partition mode requires strict CPU TCP, zero setup shared bytes, "
        "MC_NOF_BACKEND=none, no hot cache, and dynamic L1 registration");
  }
  for (const char* key : {"ipc_socket_path", "ssd_offload_path"}) {
    auto found = config_.find(key);
    if (found != config_.end() && !found->second.empty()) {
      throw std::invalid_argument("Partition mode forbids IPC and file mounts");
    }
  }
  auto offload = config_.find("enable_ssd_offload");
  if (offload != config_.end() && offload->second != "false" &&
      offload->second != "0") {
    throw std::invalid_argument("Partition mode forbids SSD offload");
  }
#endif
}

void MooncakeConnector::refresh_partition_status() {
  PartitionStatus snapshot;
  snapshot.closing = partition_closing_;
  snapshot.allocation_granularity_bytes = facebook::cachelib::Slab::kSize;
  for (const auto& [id, buffer] : local_buffers_)
    snapshot.local_buffers.push_back(buffer);
#ifdef MOONCAKE_FENCED_TCP_OWNERSHIP_API
  const auto owned = client_->GetFencedTcpOwnershipSnapshot();
  snapshot.enabled = partition_controls_ && owned.version == 1 && owned.enabled;
  snapshot.closed = owned.closed;
  snapshot.scratch_bytes = owned.scratch_bytes;
  snapshot.retained_failed_bytes = owned.failed_allocation_bytes;
  for (const auto& segment : owned.allocated_segments) {
    snapshot.shared_segments.push_back(
        {segment.id, segment.base, segment.bytes});
    snapshot.shared_allocated_bytes += segment.bytes;
  }
#endif
  std::lock_guard<std::mutex> guard(status_mutex_);
  partition_status_ = std::move(snapshot);
}

PartitionControlResult MooncakeConnector::register_l1_buffer(
    std::uintptr_t base, size_t size) {
  std::unique_lock<std::shared_mutex> access(sdk_mutex_, std::try_to_lock);
  if (!access.owns_lock()) return control_result(PartitionControlCode::BUSY);
  const auto admission = control_admission();
  if (admission != PartitionControlCode::SUCCESS)
    return control_result(admission);
  if (!base || !size ||
      size > std::numeric_limits<std::uintptr_t>::max() - base) {
    return control_result(PartitionControlCode::INVALID);
  }
  for (const auto& [id, buffer] : local_buffers_) {
    if (base == buffer.base && size == buffer.size &&
        buffer.state == LocalBufferState::REGISTERED) {
      auto result = control_result(PartitionControlCode::SUCCESS);
      result.local_buffer_id = id;
      return result;
    }
    if (base < buffer.base + buffer.size && buffer.base < base + size) {
      return control_result(PartitionControlCode::INVALID);
    }
  }
  for (const auto& segment : partition_status().shared_segments) {
    if (base < segment.base + segment.size && segment.base < base + size) {
      return control_result(PartitionControlCode::INVALID);
    }
  }
  const uint64_t id = next_registration_id_++;
  local_buffers_.emplace(
      id, LocalBufferRegistration{id, base, size, LocalBufferState::RETAINED});
  const int rc = control_sdk_call([&] {
    return client_->register_buffer(reinterpret_cast<void*>(base), size);
  });
  if (rc == 0) local_buffers_.at(id).state = LocalBufferState::REGISTERED;
  refresh_partition_status();
  auto result = control_result(
      rc == 0 ? PartitionControlCode::SUCCESS : PartitionControlCode::SDK_ERROR,
      rc);
  result.local_buffer_id = id;
  return result;
}

PartitionControlResult MooncakeConnector::unregister_l1_buffer(uint64_t id) {
  std::unique_lock<std::shared_mutex> access(sdk_mutex_, std::try_to_lock);
  if (!access.owns_lock()) return control_result(PartitionControlCode::BUSY);
  const auto admission = control_admission();
  if (admission != PartitionControlCode::SUCCESS)
    return control_result(admission);
  auto found = local_buffers_.find(id);
  if (found == local_buffers_.end())
    return control_result(PartitionControlCode::INVALID);
  found->second.state = LocalBufferState::RETAINED;
  const int rc = control_sdk_call([&] {
    return client_->unregister_buffer(
        reinterpret_cast<void*>(found->second.base));
  });
  if (rc == 0) local_buffers_.erase(found);
  refresh_partition_status();
  auto result = control_result(
      rc == 0 ? PartitionControlCode::SUCCESS : PartitionControlCode::SDK_ERROR,
      rc);
  result.local_buffer_id = id;
  return result;
}

PartitionControlResult MooncakeConnector::allocate_shared_segment(size_t size) {
  std::unique_lock<std::shared_mutex> access(sdk_mutex_, std::try_to_lock);
  if (!access.owns_lock()) return control_result(PartitionControlCode::BUSY);
  const auto admission = control_admission();
  if (admission != PartitionControlCode::SUCCESS)
    return control_result(admission);
  const size_t alignment = facebook::cachelib::Slab::kSize;
  if (!size || size > std::numeric_limits<size_t>::max() - (alignment - 1)) {
    return control_result(PartitionControlCode::INVALID);
  }
  {
    std::lock_guard<std::mutex> guard(status_mutex_);
    partition_status_.pending_shared_bytes =
        ((size + alignment - 1) / alignment) * alignment;
  }
  std::vector<std::string> ids;
  const int rc = control_sdk_call(
      [&] { return client_->allocateAndMountSegment(size, "tcp", "*", ids); });
  refresh_partition_status();
  auto result = control_result(
      rc == 0 ? PartitionControlCode::SUCCESS : PartitionControlCode::SDK_ERROR,
      rc);
  result.segment_ids = std::move(ids);
  return result;
}

PartitionControlResult MooncakeConnector::release_shared_segment(
    const std::string& id) {
  std::unique_lock<std::shared_mutex> access(sdk_mutex_, std::try_to_lock);
  if (!access.owns_lock()) return control_result(PartitionControlCode::BUSY);
  const auto admission = control_admission();
  if (admission != PartitionControlCode::SUCCESS)
    return control_result(admission);
  const auto before = partition_status();
  const auto found =
      std::find_if(before.shared_segments.begin(), before.shared_segments.end(),
                   [&](const auto& segment) { return segment.id == id; });
  if (found == before.shared_segments.end())
    return control_result(PartitionControlCode::INVALID);
  const int rc =
      control_sdk_call([&] { return client_->unmountAndFreeSegment({id}, 0); });
  refresh_partition_status();
  auto result = control_result(
      rc == 0 ? PartitionControlCode::SUCCESS : PartitionControlCode::SDK_ERROR,
      rc);
  result.segment_ids = {id};
  result.released_shared_bytes =
      before.shared_allocated_bytes - result.snapshot.shared_allocated_bytes;
  return result;
}

PartitionControlResult MooncakeConnector::close_partition() {
  if (!partition_controls_)
    return control_result(PartitionControlCode::UNSUPPORTED);
  std::unique_lock<std::mutex> closing(close_mutex_, std::try_to_lock);
  if (!closing.owns_lock()) return control_result(PartitionControlCode::BUSY);
  std::unique_lock<std::shared_mutex> access(sdk_mutex_, std::try_to_lock);
  if (!access.owns_lock()) return control_result(PartitionControlCode::BUSY);
  if (partition_status().closed)
    return control_result(PartitionControlCode::SUCCESS);
  partition_closing_ = true;
  refresh_partition_status();
  access.unlock();
  ConnectorBase<WorkerMooncakeConn>::close();
  access.lock();
  const auto before = partition_status();
  const int rc = control_sdk_call([&] { return client_->tearDownAll(); });
  if (rc == 0) local_buffers_.clear();
  refresh_partition_status();
  auto result = control_result(
      rc == 0 ? PartitionControlCode::SUCCESS : PartitionControlCode::SDK_ERROR,
      rc);
  result.released_shared_bytes = before.shared_allocated_bytes +
                                 before.retained_failed_bytes -
                                 result.snapshot.shared_allocated_bytes -
                                 result.snapshot.retained_failed_bytes;
  return result;
}

}  // namespace connector
}  // namespace lmcache
