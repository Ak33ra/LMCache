#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U0: native CPU baseline, MP roundtrip, and dependency metadata.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

run "$python" - <<'PYCODE'
import sys
import torch
import pytest
import lmcache
import lmcache.lmcache_native as native
print(sys.version)
print('torch', torch.__version__, 'pytest', pytest.__version__)
print('backend', lmcache.torch_device_type)
print(lmcache.__file__, native.__file__)
assert lmcache.torch_device_type == 'cpu'
assert native.__file__.endswith('.so')
PYCODE
run "$python" -m pytest --confcutdir=tests/v1 \
    -o addopts='' -o log_cli=false -q -ra \
    tests/v1/distributed/test_l1_manager.py \
    tests/v1/distributed/test_l1_memory_manager.py \
    tests/v1/distributed/test_store_controller.py \
    tests/v1/distributed/test_prefetch_controller.py \
    tests/v1/distributed/test_store_policy.py \
    tests/v1/distributed/test_prefetch_policy.py \
    tests/v1/multiprocess/test_session.py \
    tests/v1/multiprocess/test_tensor_wire.py

run timeout 120s "$python" examples/multi_process/cpu_inline_roundtrip.py
run uv pip check --python "$python"
