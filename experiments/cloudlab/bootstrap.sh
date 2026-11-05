#!/usr/bin/env bash
set -euo pipefail
release=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$release"
exec > >(tee -a bootstrap.log) 2>&1
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]]
[[ $(uname -m) == x86_64 ]]
jobs=${KVC_BUILD_JOBS:-8}
[[ "$jobs" =~ ^[1-8]$ ]]
unset LD_LIBRARY_PATH MC_TCP_REQUIRE_GENERATION MC_NOF_BACKEND
export CUDA_VISIBLE_DEVICES='' LMCACHE_TRACK_USAGE=false MAX_JOBS="$jobs"
export DEBIAN_FRONTEND=noninteractive
sudo -n apt-get update
sudo -n apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git ca-certificates curl pkg-config \
    python3.12-dev python3.12-venv libyaml-cpp-dev libgoogle-glog-dev \
    libgflags-dev libboost-dev libjsoncpp-dev libibverbs-dev libnuma-dev \
    libxxhash-dev libzstd-dev liburing-dev libunwind-dev \
    libcurl4-openssl-dev libmsgpack-cxx-dev libssl-dev iproute2
if [[ ! -d LMCache ]]; then
    git clone LMCache.bundle LMCache
fi
expected_head=$(python3 -c \
    'import json; print(json.load(open("source-manifest.json"))["lmcache_head"])')
[[ $(git -C LMCache rev-parse HEAD) == "$expected_head" ]]
[[ -z $(git -C LMCache status --porcelain --untracked-files=no) ]]
python3.12 -m venv venv
python_bin="$release/venv/bin/python"
"$python_bin" -m pip install 'pip==26.0.1'
"$python_bin" -m pip install -r experiments/cloudlab/requirements.lock
NO_GPU_EXT=1 "$python_bin" -m pip install -e "$release/LMCache" \
    --no-build-isolation --no-deps
KVC_MOONCAKE_SOURCE="$release/sdk" KVC_PYTHON="$python_bin" \
    KVC_BUILD_JOBS="$jobs" bash experiments/mooncake/build_fenced_sdk.sh
"$python_bin" LMCache/csrc/storage_backends/mooncake/build_partition_connector.py \
    --sdk-root "$release/sdk"
export PYTHONPATH="$release/sdk/build/python:$release/LMCache"
"$python_bin" -m pip check
"$python_bin" -m pip freeze > packages.txt
dpkg-query -W > system-packages.txt
"$python_bin" - experiments/cloudlab/bootstrap.sh <<'PY'
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import torch
import lmcache.lmcache_native as common
import lmcache.lmcache_mooncake as native
from mooncake.store import MooncakeDistributedStore

root = pathlib.Path.cwd()
assert torch.__version__ == '2.13.0+cpu', torch.__version__
assert native.partition_control_api_version == 1
files = [pathlib.Path(common.__file__), pathlib.Path(native.__file__),
         root / 'sdk/build/mooncake-store/src/mooncake_master',
         root / 'packages.txt', root / 'system-packages.txt']
files += sorted((root / 'sdk/build').rglob('*.so'))
manifest = {
    'python': sys.version,
    'platform': platform.platform(),
    'torch': str(torch.__version__),
    'partition_control_api_version': native.partition_control_api_version,
    'sdk_tree': subprocess.check_output(
        ['git', '-C', 'sdk', 'rev-parse', 'HEAD^{tree}'], text=True).strip(),
    'compiler': subprocess.check_output(['g++', '--version'], text=True),
    'artifacts': {str(p.relative_to(root)): hashlib.sha256(
        p.read_bytes()).hexdigest() for p in files},
    'source_manifest_sha256': hashlib.sha256(
        (root / 'source-manifest.json').read_bytes()).hexdigest(),
}
(root / 'build-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(root / 'READY').write_text(manifest['source_manifest_sha256'] + '\n')
PY
