#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U5b.2: build an isolated API 1 connector and run normal/fail-mount native probes.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

export MC_TCP_REQUIRE_GENERATION=1
export MC_NOF_BACKEND=none
run "$python" csrc/storage_backends/mooncake/build_partition_connector.py \
    --sdk-root "$sdk_root" --output-dir "$run_dir/native"
for mode in normal fail-mount; do
    args=()
    if [[ $mode == fail-mount ]]; then
        args+=(--fail-mount)
    fi
    run "$python" - "$run_dir/native" \
        "$lmcache_root/csrc/storage_backends/mooncake/verify_partition_controls.py" \
        --master-bin "$master_bin" --output "$run_dir/$mode" \
        "${args[@]}" <<'PYCODE'
import runpy
import sys
from pathlib import Path
import torch
import lmcache
native_dir, probe = sys.argv[1:3]
lmcache.__path__.insert(0, native_dir)
import lmcache.lmcache_mooncake as connector
print('Probe connector:', connector.__file__)
assert Path(connector.__file__).parent == Path(native_dir)
assert connector.partition_control_api_version == 1
sys.argv = [probe, *sys.argv[3:]]
runpy.run_path(probe, run_name='__main__')
PYCODE
    run "$python" -m json.tool "$run_dir/$mode/summary.json"
done
