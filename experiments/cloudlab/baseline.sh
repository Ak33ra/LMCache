#!/usr/bin/env bash
set -euo pipefail
release=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
output=${1:?Usage: baseline.sh FRESH_OUTPUT_DIRECTORY}
mkdir -p "$(dirname -- "$output")"
mkdir "$output"
output=$(cd "$output" && pwd)
cp "$release/source-manifest.json" "$release/build-manifest.json" \
    "$release/packages.txt" "$release/system-packages.txt" "$output/"
exec > >(tee "$output/baseline.log") 2>&1
unset LD_LIBRARY_PATH MC_TCP_REQUIRE_GENERATION MC_NOF_BACKEND
unset MC_NUM_TCP_IO_THREADS MC_TCP_BIND_ADDRESS
export CUDA_VISIBLE_DEVICES='' LMCACHE_TRACK_USAGE=false
export PYTHONPATH="$release/sdk/build/python:$release/LMCache"
export NUMBA_CACHE_DIR="$output/numba"
python_bin="$release/venv/bin/python"
cd "$release/LMCache"
"$python_bin" -m pytest --confcutdir=tests/v1 -o addopts='' \
    -o log_cli=false -q --junitxml="$output/pytest.xml" \
    tests/v1/distributed/test_l1_manager.py \
    tests/v1/distributed/test_l1_memory_manager.py \
    tests/v1/distributed/test_store_controller.py \
    tests/v1/distributed/test_prefetch_controller.py \
    tests/v1/distributed/test_store_policy.py \
    tests/v1/distributed/test_prefetch_policy.py \
    tests/v1/multiprocess/test_session.py \
    tests/v1/multiprocess/test_tensor_wire.py -ra
"$python_bin" - "$output/pytest.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET
suites = ET.parse(sys.argv[1]).getroot().iter('testsuite')
totals = {key: 0 for key in ('tests', 'failures', 'errors', 'skipped')}
for suite in suites:
    for key in totals:
        totals[key] += int(suite.attrib.get(key, 0))
if totals != dict(tests=311, failures=0, errors=0, skipped=0):
    raise SystemExit(f'Unexpected baseline outcome: {totals}')
PY
timeout 120s "$python_bin" examples/multi_process/cpu_inline_roundtrip.py
cd "$release/sdk"
build/mooncake-transfer-engine/tests/tcp_write_visibility_test \
    --gtest_output="xml:$output/tcp.xml"
build/mooncake-store/tests/transfer_task_test \
    --gtest_output="xml:$output/transfer.xml"
build/mooncake-store/tests/prefetch_teardown_test \
    --gtest_output="xml:$output/prefetch.xml"
build/mooncake-store/tests/fenced_tcp_mapping_test \
    --gtest_output="xml:$output/mapping.xml"
"$python_bin" -m pip check
"$python_bin" - "$output" <<'PY'
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
output = Path(sys.argv[1])
for name in ('tcp', 'transfer', 'prefetch', 'mapping'):
    suite = ET.parse(output / f'{name}.xml').getroot()
    if int(suite.get('tests', 0)) == 0 or any(
        int(suite.get(k, 0)) for k in ('failures', 'errors', 'disabled')):
        raise SystemExit(f'Incomplete native suite: {name}')
    if list(suite.iter('skipped')):
        raise SystemExit(f'Skipped native test: {name}')
(output / 'outcome.json').write_text(json.dumps({
    'outcome': 'PASSED', 'baseline_tests': 311,
    'scope': 'host-local CPU baseline and native SDK tests',
}, indent=2) + '\n')
PY
