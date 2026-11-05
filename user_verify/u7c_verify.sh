#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U7c: two-host private/shared correctness and byte-checked workload.'
source "$(dirname -- "${BASH_SOURCE[0]}")/cloudlab_common.sh"
run "${cli[@]}" run --run-id "$run_id" --output-dir "$run_dir/partition" --workload
run "$python" -m json.tool "$run_dir/partition/outcome.json"
