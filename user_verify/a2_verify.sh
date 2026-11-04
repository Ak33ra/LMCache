#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='A2: Store-owned placement and quiescent lending with the installed wheel.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

store_owned_root=${KVC_STORE_OWNED_ROOT:-/scratch/akira/kvc-worktrees/store-owned}
cd -- "$store_owned_root"
export PYTHONPATH="$PWD"
wheel_environment
run "$python" tools/partitioning/probe_lent_segments.py \
    --mode all --cycles 2 --output "$run_dir/all"
run "$python" -m json.tool "$run_dir/all/observations.json"
run "$python" tools/partitioning/probe_lent_segments.py \
    --mode lend --cycles 1 --grace-seconds 1 --output "$run_dir/graceful"
run "$python" -m json.tool "$run_dir/graceful/observations.json"
