#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U2: live two-MP shared copies under default and skip_l1 policies.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

cd -- "$project_root"
for policy in default skip_l1; do
    run timeout 180s "$python" experiments/mooncake/check_two_mp.py \
        --master-bin "$master_bin" --store-policy "$policy" \
        --output-dir "$run_dir/$policy"
    run "$python" -m json.tool "$run_dir/$policy/summary.json"
done
