#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U4e: inline operation lifetimes, CPU smoke, portable replay, and trace CLI.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

run "$python" -m pytest --confcutdir=tests/v1 \
    -o addopts='' -o log_cli=false -q  \
    tests/v1/multiprocess/test_inline_operation_lifetimes.py \
    tests/v1/distributed/test_l1_write_operations.py

run timeout 120s "$python" examples/multi_process/cpu_inline_roundtrip.py
run "$python" -m pytest --confcutdir=tests/v1 \
    -o addopts='' -o log_cli=false -q  \
    tests/v1/distributed/test_multi_l1.py::test_inline_operations_replay_portable_call_schema
run "$python" -m pytest --confcutdir=tests/v1 \
    -o addopts='' -o log_cli=false -q -k 'inline_write_outcome or operation_replay_tracks or portable_call_schema' \
    tests/v1/distributed/test_multi_l1.py
run "$python" -m pytest --confcutdir=tests/cli \
    -o addopts='' -o log_cli=false -q  \
    tests/cli/commands/trace/test_dispatch.py \
    tests/cli/commands/trace/test_driver.py
