#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U7b: native CPU baseline and fenced SDK suites on both hosts.'
source "$(dirname -- "${BASH_SOURCE[0]}")/cloudlab_common.sh"
run "${cli[@]}" baseline --run-id "$run_id" --output-dir "$run_dir/baseline"
run "$python" -m json.tool "$run_dir/baseline/outcome.json"
