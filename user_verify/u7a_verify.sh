#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U7a: package clean source and build both fresh Ubuntu CPU hosts.'
source "$(dirname -- "${BASH_SOURCE[0]}")/cloudlab_common.sh"
run "${cli[@]}" setup --output-dir "$run_dir/setup"
run "$python" -m json.tool "$run_dir/setup/outcome.json"
