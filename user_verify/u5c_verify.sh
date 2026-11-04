#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
description='U5c: live partition endpoints, same-process resize, charge and delay inspection.'
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

cd -- "$project_root"
run "$python" experiments/partitioning/partition_roundtrip.py \
    --lmcache-root "$lmcache_root" --master-bin "$master_bin" \
    --output-dir "$run_dir/partition"

run "$python" -m json.tool "$run_dir/partition/correctness.json"
run "$python" -m json.tool "$run_dir/partition/summary.json"
run "$python" - "$run_dir/partition" <<'PYCODE'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
rows = [json.loads(p.read_text()) for p in root.glob('*-partition-*.json')]
assert rows, 'Missing partition charge samples'
print('samples', len(rows))
print('peak managed MiB', max(r['charged_bytes'] for r in rows) / 2**20)
correctness = json.loads((root / 'correctness.json').read_text())
summary = json.loads((root / 'summary.json').read_text())
assert summary['success'] is True
print('B PIDs', sorted({r['pid'] for r in rows
                        if r['pid'] == correctness['same_pid']}))
assert all(r['charged_bytes'] <= 88 * 2**20 for r in rows)
timings = sorted(root.glob('*-resize-timing.json'))
assert timings, 'Missing resize timing records'
for path in timings:
    result = json.loads(path.read_text())
    print(result['request_id'], round(result['elapsed_ns'] / 1e6, 2),
          'ms observed')
PYCODE
