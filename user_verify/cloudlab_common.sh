#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help ]]; then
    printf '%s\n' "$description"
    printf 'Set KVC_CLOUDLAB_INVENTORY; optional KVC_CLOUDLAB_SSH_CONFIG.\n'
    printf 'U7d also requires KVC_CLOUDLAB_RUN_ID for the existing run.\n'
    exit 0
fi
(( $# == 0 )) || { printf 'Use --help; arguments are not supported.\n' >&2; exit 2; }
project_root=$(realpath -- "${KVC_PROJECT_ROOT:-$script_dir/..}")
python=${KVC_CLOUDLAB_PYTHON:-python3}
inventory=${KVC_CLOUDLAB_INVENTORY:?Set KVC_CLOUDLAB_INVENTORY to your JSON inventory}
results_root=${KVC_RESULTS_ROOT:-$project_root/user_verify/results}
mkdir -p -- "$results_root"
run_dir=$(mktemp -d "$results_root/$(basename -- "$0" .sh).XXXXXXXX")
run_id=${KVC_CLOUDLAB_RUN_ID:-$(basename -- "$run_dir")}
exec > >(tee "$run_dir/run.log") 2>&1
trap 'status=$?; printf "Exit status: %s\nResults: %s\n" "$status" "$run_dir"' EXIT
cli=("$python" "$project_root/experiments/cloudlab/cloudlab.py"
     --inventory "$inventory" --project-root "$project_root"
     --lmcache-root "${KVC_LMCACHE_ROOT:-$project_root/LMCache}")
if [[ -n ${KVC_CLOUDLAB_SSH_CONFIG:-} ]]; then
    cli+=(--ssh-config "$KVC_CLOUDLAB_SSH_CONFIG")
fi
printf '%s\nResults: %s\nRun ID: %s\n' "$description" "$run_dir" "$run_id"
run()
{
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    "$@"
}
