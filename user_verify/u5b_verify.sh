#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U5b: controlled suites, isolated native build/probes, public probes, and strict docs.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

export KVC_RESULTS_ROOT="$run_dir"
run "$script_dir/u5b_verify_1.sh"
run "$script_dir/u5b_verify_2.sh"
run "$script_dir/u5b_verify_3.sh"
