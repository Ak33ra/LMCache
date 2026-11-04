#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U4b: allocation-bound read lifetimes.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

run "$python" -m pytest --confcutdir=tests/v1 \
    -o addopts='' -o log_cli=false -q  \
    tests/v1/distributed/test_l1_read_operations.py \
    tests/v1/distributed/test_l1_manager.py
