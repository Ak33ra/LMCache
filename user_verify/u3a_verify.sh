#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U3a: removable DRAM tests and held-view mapping example.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

run "$python" -m pytest --confcutdir=tests/v1 \
    -o addopts='' -o log_cli=false -q  \
    tests/v1/distributed/test_segmented_dram.py \
    tests/v1/distributed/test_segmented_dram_l1.py

run "$python" - <<'PYCODE'
import mmap
from pathlib import Path
import torch
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.memory_manager.segmented_dram import (
    SegmentedDRAML1MemoryManager,
)
pool = SegmentedDRAML1MemoryManager()
segment = pool.add_segment(mmap.PAGESIZE)
layout = MemoryLayoutDesc([torch.Size([mmap.PAGESIZE])], [torch.uint8])
error, objects = pool.allocate(layout, 1)
view = objects[0].raw_tensor
view.fill_(73)
pool.begin_drain(segment.handle)
pool.free(objects)
print('registration range:', hex(segment.base), segment.mapped_bytes)
print('held view:', pool.remove_segment(segment.handle))
print('usage:', pool.get_memory_usage())
assert pool.get_memory_usage() == (0, mmap.PAGESIZE)
assert torch.all(view == 73)
del view
print('released view:', pool.remove_segment(segment.handle))
print('usage:', pool.get_memory_usage())
assert pool.get_memory_usage() == (0, 0)
assert segment.mapping_name not in Path('/proc/self/maps').read_text()
pool.close()
PYCODE
