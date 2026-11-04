#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U4d.1: immediate and graceful fenced TCP handoff.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

export PYTHONPATH="$sdk_root/build/python:$lmcache_root"
export MC_TCP_REQUIRE_GENERATION=1
export MC_NOF_BACKEND=none
cd -- "$project_root"
for grace in 0 1; do
    run "$python" experiments/mooncake/check_fenced_handoff.py \
        --grace-seconds "$grace" --output "$run_dir/grace-$grace"
    run "$python" -m json.tool "$run_dir/grace-$grace/summary.json"
done
