#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U3b: three physical handoff cycles with a persistent L1 and SDK wheel.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

wheel_environment
cd -- "$project_root"
run "$python" experiments/mooncake/check_segment_handoff.py \
    --cycles 3 --output "$run_dir/handoff"
run "$python" -m json.tool "$run_dir/handoff/summary.json"
