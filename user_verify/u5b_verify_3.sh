#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U5b.3: public activation probes and strict Sphinx build.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

cd -- "$project_root"
run "$python" -m pytest --noconftest -o addopts='' -o log_cli=false -q \
    experiments/partitioning/results/2026-11-03-review-probes/activation_probes.py
cd -- "$lmcache_root"
sphinx_build=${KVC_SPHINX_BUILD:-/scratch/akira/kvc-worktrees/dram-segments-docs-env/bin/sphinx-build}
run "$sphinx_build" -b html -W --keep-going docs/source "$run_dir/docs-html"
