#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='Optional U1 rebuild: fresh strict SDK plus isolated connector; leaves main extension intact.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

export KVC_PYTHON="$python"
export KVC_BUILD_JOBS="$MAX_JOBS"
export KVC_MOONCAKE_SOURCE="$run_dir/mooncake-fenced"
cd -- "$project_root"
run bash experiments/mooncake/build_fenced_sdk.sh
cd -- "$lmcache_root"
run "$python" csrc/storage_backends/mooncake/build_partition_connector.py \
    --sdk-root "$KVC_MOONCAKE_SOURCE" --output-dir "$run_dir/native"
printf 'Keep %s: the isolated connector links to this SDK build.\n' \
    "$KVC_MOONCAKE_SOURCE"
