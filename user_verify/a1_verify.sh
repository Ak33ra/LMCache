#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='A1: isolated common-arena alternative and 21 local tests.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

arena_root=${KVC_ARENA_ROOT:-/scratch/akira/kvc-worktrees/common-arena}
cd -- "$arena_root"
export PYTHONPATH="$PWD"
run "$python" -m examples.partitioning_common_arena.verify
run "$python" -m pytest --noconftest -o addopts='' -o log_cli=false -q \
    examples/partitioning_common_arena/test_arena.py
