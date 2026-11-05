#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U7d: inspect, stop and recover evidence from an existing run.'
source "$(dirname -- "${BASH_SOURCE[0]}")/cloudlab_common.sh"
: "${KVC_CLOUDLAB_RUN_ID:?Set the existing run ID to inspect and stop}"
status=0
run "${cli[@]}" status --run-id "$run_id" || status=$?
run "${cli[@]}" stop --run-id "$run_id" || status=$?
run "${cli[@]}" collect --run-id "$run_id" \
    --output-dir "$run_dir/collected" || status=$?
exit "$status"
