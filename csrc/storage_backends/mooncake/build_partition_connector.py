# SPDX-License-Identifier: Apache-2.0
"""Build only this checkout's Mooncake extension against an existing CPU SDK."""

# Standard
from pathlib import Path
import argparse
import shutil
import sysconfig

# Third Party
from torch.utils.cpp_extension import load


def build(sdk: Path, output_dir: Path | None = None) -> None:
    """Compile into a local directory and copy only the Mooncake extension.

    Args:
        sdk: SDK source checkout with its existing CMake build directory.
        output_dir: Optional isolated destination for compatibility checks.
    """
    root = Path(__file__).resolve().parents[3]
    includes = [
        "mooncake-store/include",
        "mooncake-store/include/cachelib_memory_allocator",
        "mooncake-store/include/cachelib_memory_allocator/include",
        "mooncake-store/include/cachelib_memory_allocator/fake_include",
        "mooncake-transfer-engine/include",
        "mooncake-common/include",
        "build/mooncake-store/include",
        "build/mooncake-transfer-engine/include",
        "build/_deps/yalantinglibs-src/include",
        "build/_deps/yalantinglibs-src/include/ylt/thirdparty",
        "build/_deps/yalantinglibs-src/include/ylt/standalone",
    ]
    directory = root / ".audit" / ("native-controls-" + sdk.name)
    directory.mkdir(parents=True, exist_ok=True)
    library = sdk / "build/mooncake-store/src"
    module = load(
        name="lmcache_mooncake",
        sources=[
            str(Path(__file__).parent / name)
            for name in ("connector.cpp", "pybind.cpp")
        ],
        extra_include_paths=[str(sdk / path) for path in includes]
        + ["/usr/include/jsoncpp"],
        extra_cflags=["-O2", "-std=c++20", "-DYLT_ENABLE_IBV"],
        extra_ldflags=[f"-L{library}", "-lmooncake_store", f"-Wl,-rpath,{library}"],
        build_directory=str(directory),
        verbose=True,
    )
    output_dir = output_dir or root / "lmcache"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / (
        "lmcache_mooncake" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    if destination.is_symlink():
        raise RuntimeError("Refusing to overwrite a Mooncake extension symlink")
    shutil.copy2(module.__file__, destination)
    print(
        f"Built {destination}; partition API {module.partition_control_api_version}, "
        f"Store generation API {module.store_generation_binding_api_version}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    build(args.sdk_root.resolve(), args.output_dir)
