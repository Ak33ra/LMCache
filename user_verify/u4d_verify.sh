#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U4d: both live grace variants, then all four native test binaries.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

export KVC_RESULTS_ROOT="$run_dir"
run "$script_dir/u4d_verify_1.sh"
run "$script_dir/u4d_verify_2.sh"
