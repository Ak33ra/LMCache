// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lmcache {
namespace connector {

/** Outcome of one synchronous partition control attempt. */
enum class PartitionControlCode {
  SUCCESS,
  BUSY,
  INVALID,
  SDK_ERROR,
  CLOSED,
  UNSUPPORTED,
};

/** A failed registration remains charged until deregistration is confirmed. */
enum class LocalBufferState { REGISTERED, RETAINED };

/** Caller-owned CPU backing registered only for local transport access. */
struct LocalBufferRegistration {
  uint64_t id{0};
  std::uintptr_t base{0};
  size_t size{0};
  LocalBufferState state{LocalBufferState::REGISTERED};
};

/** SDK-owned shared backing advertised to the Store master. */
struct SharedSegmentAllocation {
  std::string id;
  std::uintptr_t base{0};
  size_t size{0};
};

/** Last confirmed ownership snapshot, available without waiting for SDK I/O. */
struct PartitionStatus {
  bool enabled{false};
  bool closing{false};
  bool closed{false};
  size_t scratch_bytes{0};
  size_t allocation_granularity_bytes{0};
  size_t pending_shared_bytes{0};
  size_t shared_allocated_bytes{0};
  size_t retained_failed_bytes{0};
  std::vector<LocalBufferRegistration> local_buffers;
  std::vector<SharedSegmentAllocation> shared_segments;
};

/** Result includes retained ownership even when an SDK mutation fails. */
struct PartitionControlResult {
  PartitionControlCode code{PartitionControlCode::SUCCESS};
  int sdk_error{0};
  uint64_t local_buffer_id{0};
  std::vector<std::string> segment_ids;
  size_t released_shared_bytes{0};
  PartitionStatus snapshot;
};

}  // namespace connector
}  // namespace lmcache
