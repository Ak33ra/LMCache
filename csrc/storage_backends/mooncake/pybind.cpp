// SPDX-License-Identifier: Apache-2.0
#include <stdexcept>
#include <string>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../connector_pybind_utils.h"
#include "connector.h"

namespace py = pybind11;

PYBIND11_MODULE(lmcache_mooncake, m) {
  using namespace lmcache::connector;
#ifdef MOONCAKE_FENCED_TCP_OWNERSHIP_API
  m.attr("partition_control_api_version") = MOONCAKE_FENCED_TCP_OWNERSHIP_API;
#else
  m.attr("partition_control_api_version") = 0;
#endif
#ifdef MOONCAKE_STORE_GENERATION_BINDING_API
  m.attr("store_generation_binding_api_version") =
      MOONCAKE_STORE_GENERATION_BINDING_API;
#else
  m.attr("store_generation_binding_api_version") = 0;
#endif
  py::enum_<PartitionControlCode>(m, "PartitionControlCode")
      .value("SUCCESS", PartitionControlCode::SUCCESS)
      .value("BUSY", PartitionControlCode::BUSY)
      .value("INVALID", PartitionControlCode::INVALID)
      .value("SDK_ERROR", PartitionControlCode::SDK_ERROR)
      .value("CLOSED", PartitionControlCode::CLOSED)
      .value("UNSUPPORTED", PartitionControlCode::UNSUPPORTED);
  py::enum_<LocalBufferState>(m, "LocalBufferState")
      .value("REGISTERED", LocalBufferState::REGISTERED)
      .value("RETAINED", LocalBufferState::RETAINED);
  py::class_<LocalBufferRegistration>(m, "LocalBufferRegistration")
      .def_readonly("id", &LocalBufferRegistration::id)
      .def_readonly("base", &LocalBufferRegistration::base)
      .def_readonly("size", &LocalBufferRegistration::size)
      .def_readonly("state", &LocalBufferRegistration::state);
  py::class_<SharedSegmentAllocation>(m, "SharedSegmentAllocation")
      .def_readonly("id", &SharedSegmentAllocation::id)
      .def_readonly("base", &SharedSegmentAllocation::base)
      .def_readonly("size", &SharedSegmentAllocation::size);
  py::class_<PartitionStatus>(m, "PartitionStatus")
      .def_readonly("enabled", &PartitionStatus::enabled)
      .def_readonly("closing", &PartitionStatus::closing)
      .def_readonly("closed", &PartitionStatus::closed)
      .def_readonly("scratch_bytes", &PartitionStatus::scratch_bytes)
      .def_readonly("allocation_granularity_bytes",
                    &PartitionStatus::allocation_granularity_bytes)
      .def_readonly("pending_shared_bytes",
                    &PartitionStatus::pending_shared_bytes)
      .def_readonly("shared_allocated_bytes",
                    &PartitionStatus::shared_allocated_bytes)
      .def_readonly("retained_failed_bytes",
                    &PartitionStatus::retained_failed_bytes)
      .def_readonly("local_buffers", &PartitionStatus::local_buffers)
      .def_readonly("shared_segments", &PartitionStatus::shared_segments);
  py::class_<PartitionControlResult>(m, "PartitionControlResult")
      .def_readonly("code", &PartitionControlResult::code)
      .def_readonly("sdk_error", &PartitionControlResult::sdk_error)
      .def_readonly("local_buffer_id", &PartitionControlResult::local_buffer_id)
      .def_readonly("segment_ids", &PartitionControlResult::segment_ids)
      .def_readonly("released_shared_bytes",
                    &PartitionControlResult::released_shared_bytes)
      .def_readonly("snapshot", &PartitionControlResult::snapshot);

  py::class_<lmcache::connector::L1RegistrationConfig>(m,
                                                       "L1RegistrationConfig")
      .def(py::init<>())
      .def_readwrite("enabled",
                     &lmcache::connector::L1RegistrationConfig::enabled)
      .def_readwrite("base", &lmcache::connector::L1RegistrationConfig::base)
      .def_readwrite("size", &lmcache::connector::L1RegistrationConfig::size);

  py::class_<lmcache::connector::MooncakeConnector>(m, "LMCacheMooncakeClient")
      .def(py::init([](lmcache::connector::ConfigDict config, int num_workers,
                       lmcache::connector::L1RegistrationConfig l1_registration,
                       py::object per_op_workers, bool partition_controls) {
             return new lmcache::connector::MooncakeConnector(
                 std::move(config), num_workers, l1_registration,
                 lmcache::connector::pybind_utils::parse_per_op_workers(
                     per_op_workers),
                 partition_controls);
           }),
           py::arg("config"), py::arg("num_workers"),
           py::arg("l1_registration") =
               lmcache::connector::L1RegistrationConfig{},
           py::arg("per_op_workers") = py::none(),
           py::arg("partition_controls") = false)
      .def("partition_status", &MooncakeConnector::partition_status)
      .def("register_l1_buffer", &MooncakeConnector::register_l1_buffer,
           py::arg("base"), py::arg("size"),
           py::call_guard<py::gil_scoped_release>())
      .def("unregister_l1_buffer", &MooncakeConnector::unregister_l1_buffer,
           py::arg("id"), py::call_guard<py::gil_scoped_release>())
      .def("allocate_shared_segment",
           &MooncakeConnector::allocate_shared_segment, py::arg("size"),
           py::call_guard<py::gil_scoped_release>())
      .def("release_shared_segment", &MooncakeConnector::release_shared_segment,
           py::arg("id"), py::call_guard<py::gil_scoped_release>())
      .def("close_partition", &MooncakeConnector::close_partition,
           py::call_guard<py::gil_scoped_release>())
          LMCACHE_BIND_CONNECTOR_METHODS(lmcache::connector::MooncakeConnector);
}
