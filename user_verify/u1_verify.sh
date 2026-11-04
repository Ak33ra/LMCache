#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U1: inspect compiled API 1 connector and pinned fenced SDK revision.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

run "$python" - <<'PYCODE'
import torch
import lmcache.lmcache_mooncake as connector
print(connector.__file__)
print('partition API', connector.partition_control_api_version)
assert connector.__file__.endswith('.so')
assert connector.partition_control_api_version == 1
PYCODE
run git -C "$sdk_root" rev-parse HEAD
revision=$(git -C "$sdk_root" rev-parse HEAD)
[[ $revision == 170e8dfc6624eeb85066d8aae3b505351f134e95 ]]
