#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U5a: capacity arithmetic tests and requested/effective target loop.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

run "$python" -m pytest --confcutdir=tests/v1 \
    -o addopts='' -o log_cli=false -q  \
    tests/v1/distributed/test_partition_config.py

run "$python" - <<'PYCODE'
from dataclasses import asdict
import json
from lmcache.v1.distributed.partition_config import PartitionConfig
mib = 1 << 20
config = PartitionConfig(88 * mib, 16 * mib, 8 * mib, 0.5)
for alpha in (0, 0.3, 0.5, 1):
    print(json.dumps({
        'alpha': alpha, **asdict(config.capacity(16 * mib, alpha=alpha)),
    }))
PYCODE
