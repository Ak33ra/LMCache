#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U4d.2: four native fenced TCP suites; one test waits 61 seconds.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

export MC_TCP_REQUIRE_GENERATION=1
export MC_NOF_BACKEND=none
cd -- "$sdk_root"
run build/mooncake-transfer-engine/tests/tcp_write_visibility_test
run build/mooncake-store/tests/transfer_task_test --gtest_filter='*Fenced*'
run build/mooncake-store/tests/prefetch_teardown_test
run build/mooncake-store/tests/fenced_tcp_mapping_test
