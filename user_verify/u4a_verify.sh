#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U4a: reproduce the known TTL gap and verify the expected failing safety check.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

cd -- "$project_root"
run "$python" experiments/mooncake/check_expired_access.py
status=0
run "$python" experiments/mooncake/check_expired_access.py \
    --require-safe > "$run_dir/require-safe.log" 2>&1 || status=$?
cat "$run_dir/require-safe.log"
if (( status == 0 )); then
    printf 'Expected --require-safe to fail; the documented counterexample changed.\n' >&2
    exit 1
fi
run grep -Fx 'AssertionError: TTL expiry allowed reuse before reader completion' \
    "$run_dir/require-safe.log"
printf 'EXPECTED COUNTEREXAMPLE: --require-safe exited %s with the TTL diagnostic.\n' \
    "$status"
