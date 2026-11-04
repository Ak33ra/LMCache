#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U5b.1: admission, callers, staged ownership, capabilities, BUSY, RPC, activation, adapters.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

run "$python" -m pytest --confcutdir=tests/v1 \
    -o addopts='' -o log_cli=false -q  \
    tests/v1/distributed/test_l1_admission.py \
    tests/v1/distributed/test_l1_write_operations.py \
    tests/v1/distributed/test_l1_read_operations.py \
    tests/v1/distributed/test_l1_interface.py

run "$python" -m pytest --confcutdir=tests/v1 \
    -o addopts='' -o log_cli=false -q -ra \
    tests/v1/distributed/test_l1_manager.py \
    tests/v1/distributed/test_prefetch_controller.py \
    tests/v1/distributed/test_prefetch_write_operations.py \
    tests/v1/distributed/test_multi_l1.py \
    tests/v1/distributed/test_eviction_extra_logging.py \
    tests/v1/multiprocess/test_inline_operation_lifetimes.py \
    tests/v1/multiprocess/test_engine_driven_multigroup.py

run "$python" -m pytest --confcutdir=tests/v1/distributed \
    -o addopts='' -o log_cli=false -q  \
    tests/v1/distributed/test_store_staged_admission.py \
    tests/v1/distributed/test_store_read_operations.py \
    tests/v1/distributed/test_store_controller.py \
    tests/v1/distributed/test_runtime_adapters.py

run "$python" -m pytest --confcutdir=tests/v1 \
    -o addopts='' -o log_cli=false -q  \
    tests/v1/distributed/test_store_staged_admission.py \
    tests/v1/distributed/test_l1_write_operations.py \
    tests/v1/distributed/test_store_read_operations.py \
    tests/v1/distributed/test_store_controller.py \
    tests/v1/distributed/test_runtime_adapters.py \
    tests/v1/distributed/test_l1_admission.py \
    tests/v1/distributed/test_prefetch_write_operations.py


run "$python" -m pytest --noconftest -o addopts='' -o log_cli=false -q \
    tests/v1/distributed/test_partition_manager.py
run "$python" -m pytest --confcutdir=tests/v1 \
    -o addopts='' -o log_cli=false -q  \
    tests/v1/multiprocess/test_request_limits.py

run "$python" -m pytest --confcutdir=tests/v1 \
    -o addopts='' -o log_cli=false -q  \
    tests/v1/distributed/test_partition_config.py \
    tests/v1/distributed/test_partition_startup_config.py \
    tests/v1/distributed/test_partition_manager.py \
    tests/v1/distributed/test_prefetch_write_operations.py \
    tests/v1/multiprocess/test_inline_operation_lifetimes.py \
    tests/v1/multiprocess/test_request_limits.py

run "$python" -m pytest --confcutdir=tests/v1 \
    -o addopts='' -o log_cli=false -q -k 'not Integration' -ra \
    tests/v1/distributed/test_native_connector_l2_adapter.py \
    tests/v1/distributed/test_mooncake_store_l2_adapter.py
