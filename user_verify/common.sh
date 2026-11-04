#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
step=$(basename -- "$0" .sh)
if [[ ${1:-} == --help ]]; then
    printf 'Usage: %s\n\n%s\n' "$0" "$description"
    printf 'No arguments. Set KVC_* overrides from %s/README.md.\n' "$script_dir"
    exit 0
fi
if (( $# )); then
    printf 'Unexpected argument: %s (use --help)\n' "$1" >&2
    exit 2
fi

project_root=$(realpath -- "${KVC_PROJECT_ROOT:-$script_dir/..}")
lmcache_root=$(realpath -- "${KVC_LMCACHE_ROOT:-$project_root/LMCache}")
sdk_root=$(realpath -- \
    "${KVC_SDK_ROOT:-/scratch/akira/kvc-worktrees/mooncake-fenced}")
python=${KVC_PYTHON:-/scratch/akira/lmcache-dev/bin/python}
python=$(command -v -- "$python")
master_bin=${KVC_MASTER_BIN:-$sdk_root/build/mooncake-store/src/mooncake_master}
results_root=${KVC_RESULTS_ROOT:-$project_root/user_verify/results}
mkdir -p -- "$results_root"
results_root=$(realpath -- "$results_root")
export KVC_PROJECT_ROOT="$project_root"
export KVC_LMCACHE_ROOT="$lmcache_root"
export KVC_SDK_ROOT="$sdk_root"
export KVC_PYTHON="$python"
run_dir=$(mktemp -d "$results_root/$step.XXXXXXXX")
exec > >(tee "$run_dir/run.log") 2>&1
trap 'status=$?; printf "Exit status: %s\nResults: %s\n" \
    "$status" "$run_dir"' EXIT

unset LD_LIBRARY_PATH MC_TCP_REQUIRE_GENERATION MC_NOF_BACKEND
unset PYTHONOPTIMIZE PYTEST_ADDOPTS
export PYTHONPATH="$lmcache_root"
export CUDA_VISIBLE_DEVICES=''
export LMCACHE_TRACK_USAGE=false
export NUMBA_CACHE_DIR="$run_dir/numba"
export UV_CACHE_DIR="${KVC_UV_CACHE_DIR:-/tmp/kvc-uv-cache}"
export MAX_JOBS="${KVC_BUILD_JOBS:-8}"
if [[ ! $MAX_JOBS =~ ^[1-8]$ ]]; then
    printf 'KVC_BUILD_JOBS must be an integer from 1 through 8.\n' >&2
    exit 2
fi
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export PATH="$(dirname -- "$python"):$PATH"

printf '%s\nResults: %s\nPython: %s\nLMCache: %s\nSDK: %s\n' \
    "$description" "$run_dir" "$python" "$lmcache_root" "$sdk_root"

## Run a concrete command while preserving its status and quoted arguments.
run()
{
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    "$@"
}

## Select only the installed wheel's required CUDA runtime library directory.
wheel_environment()
{
    export LD_LIBRARY_PATH
    LD_LIBRARY_PATH=$("$python" -c \
        'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cuda_runtime/lib")')
    printf 'Wheel runtime library directory: %s\n' "$LD_LIBRARY_PATH"
}

cd -- "$lmcache_root"
run "$python" --version
run git -C "$lmcache_root" rev-parse HEAD
run git -C "$sdk_root" rev-parse HEAD
