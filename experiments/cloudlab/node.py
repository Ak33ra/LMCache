# SPDX-License-Identifier: Apache-2.0
"""Host-local checks, coordinator and recovery for a packaged experiment."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import runpy
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "partitioning"))
from process_host import (
    ProcessHost,
    group_members,
    identity,
    matches,
    recover,
    write_json,
)


def verify_release(release: Path) -> dict:
    """Reject partial builds, modified packages and changed native artifacts."""
    manifest_path = release / "source-manifest.json"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if (release / "READY").read_text().strip() != digest:
        raise RuntimeError("Release has no matching completed build")
    source = json.loads(manifest_path.read_text())
    build = json.loads((release / "build-manifest.json").read_text())
    if build["source_manifest_sha256"] != digest:
        raise RuntimeError("Build/source manifest mismatch")
    for name, expected in {**source["files"], **build["artifacts"]}.items():
        if (
            hashlib.sha256((release / name).read_bytes()).hexdigest()
            != expected
        ):
            raise RuntimeError(f"Changed release file: {name}")
    head = subprocess.check_output(
        ["git", "-C", str(release / "LMCache"), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(release / "LMCache"), "diff", "HEAD"], text=True
    )
    if head != source["lmcache_head"] or dirty:
        raise RuntimeError("Changed LMCache source")
    sdk_tree = subprocess.check_output(
        ["git", "-C", str(release / "sdk"), "rev-parse", "HEAD^{tree}"],
        text=True,
    ).strip()
    sdk_dirty = subprocess.check_output(
        ["git", "-C", str(release / "sdk"), "diff", "HEAD"], text=True
    )
    if sdk_tree != build["sdk_tree"] or sdk_dirty:
        raise RuntimeError("Changed SDK source")
    if (
        build["sdk_tree"]
        != (release / "experiments/mooncake/sdk-patches/expected-tree.txt")
        .read_text()
        .strip()
    ):
        raise RuntimeError("Unexpected SDK tree")
    freeze = subprocess.check_output(
        [str(release / "venv/bin/python"), "-m", "pip", "freeze"], text=True
    )
    if freeze != (release / "packages.txt").read_text():
        raise RuntimeError(
            "Installed Python packages differ from build manifest"
        )
    return build


def check(release: Path, inventory: dict, host: str) -> dict:
    """Verify installed binaries, experiment address ownership and unused ports."""
    build = verify_release(release)
    address = inventory[host]["experiment_ip"]
    ports = range(inventory["port_base"], inventory["port_base"] + 9)
    owned_ports = list(ports)[:6] if host == "node0" else list(ports)[6:]
    sockets = []
    try:
        for port in owned_ports:
            probe = socket.socket()
            sockets.append(probe)
            probe.bind((address, port))
    finally:
        for probe in sockets:
            probe.close()
    peer = inventory["node1" if host == "node0" else "node0"]["experiment_ip"]
    return dict(
        host=socket.gethostname(),
        build=build,
        address=address,
        available_ports=owned_ports,
        addresses=subprocess.check_output(["ip", "-j", "address"], text=True),
        route=subprocess.check_output(
            ["ip", "-j", "route", "get", peer], text=True
        ),
        listeners=subprocess.check_output(["ss", "-ltnp"], text=True),
    )


def peer_probe(release: Path, inventory: dict) -> None:
    """Require node0 peer SSH and a real data-LAN connection before services."""
    peer = inventory["node1"]
    command = [
        str(release / "venv/bin/python"),
        str(Path(__file__).resolve()),
        "listen",
        "--address",
        peer["experiment_ip"],
        "--port",
        "0",
    ]
    with subprocess.Popen(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            peer["ssh"],
            "exec " + shlex.join(command),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as listener:
        line = listener.stdout.readline()
        if not line.startswith("READY "):
            raise RuntimeError(
                "node0-to-node1 SSH/listener preflight failed: "
                + listener.stderr.read()
            )
        with socket.create_connection(
            (peer["experiment_ip"], int(line.split()[1])),
            timeout=10,
            source_address=(inventory["node0"]["experiment_ip"], 0),
        ) as connection:
            if connection.recv(64) != b"kvc-experiment-network":
                raise RuntimeError(
                    "Experiment network probe returned wrong data"
                )
        if listener.wait(timeout=20):
            raise RuntimeError("Peer network probe failed")


def coordinate(
    release: Path,
    directory: Path,
    inventory: dict,
    workload: bool,
    baseline: bool = False,
) -> int:
    """Run existing assertions and keep cleanup active after workstation EOF."""
    log = (directory / "coordinator.log").open("a", buffering=1)
    os.dup2(log.fileno(), sys.stdout.fileno())
    os.dup2(log.fileno(), sys.stderr.fileno())
    record = dict(identity=identity(os.getpid()), run_id=directory.name)
    write_json(directory / "coordinator.json", record)
    outcome = dict(outcome="BLOCKED", run_id=directory.name)

    def interrupted(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    def watch_control() -> None:
        if not os.read(sys.stdin.fileno(), 1):
            os.kill(os.getpid(), signal.SIGTERM)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    threading.Thread(target=watch_control, daemon=True).start()
    try:
        if baseline:
            verify_release(release)
            outcome["outcome"] = "FAILED"
            host = ProcessHost(release, directory / "services")
            command = [
                "bash",
                str(release / "experiments/cloudlab/baseline.sh"),
                str(directory / "evidence"),
            ]
            with host.service("baseline", command, directory) as process:
                while process.poll() is None:
                    time.sleep(0.2)
                if process.returncode != 0:
                    raise RuntimeError(
                        f"Baseline exited with {process.returncode}"
                    )
            baseline_result = json.loads(
                (directory / "evidence/outcome.json").read_text()
            )
            if baseline_result.get("outcome") != "PASSED":
                raise RuntimeError("Baseline script did not record PASSED")
        else:
            check(release, inventory, "node0")
            peer_probe(release, inventory)
            outcome["outcome"] = "FAILED"
            environment = dict(
                CUDA_VISIBLE_DEVICES="",
                LMCACHE_TRACK_USAGE="false",
                PYTHONPATH=f"{release}/sdk/build/python:{release}/LMCache",
            )
            os.environ.update(environment)
            os.environ.pop("LD_LIBRARY_PATH", None)
            sys.path[:0] = [
                str(release / "sdk/build/python"),
                str(release / "LMCache"),
            ]
            sys.argv = [
                str(
                    release / "experiments/partitioning/partition_roundtrip.py"
                ),
                "--lmcache-root",
                str(release / "LMCache"),
                "--master-bin",
                str(release / "sdk/build/mooncake-store/src/mooncake_master"),
                "--output-dir",
                str(directory / "evidence"),
                "--deployment",
                str(directory / "inventory.json"),
                "--peer-output",
                str(directory / "services"),
            ]
            if workload:
                sys.argv.append("--workload")
            runpy.run_path(sys.argv[0], run_name="__main__")
        outcome.update(outcome="PASSED", assertions=True)
    except (Exception, KeyboardInterrupt):
        traceback.print_exc()
        outcome["error"] = traceback.format_exc()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            cleanup = recover(directory / "evidence") + recover(
                directory / "services"
            )
        except RuntimeError as error:
            cleanup = [dict(stopped=False, escalated=False, error=str(error))]
        outcome["cleanup"] = cleanup
        if any(not item["stopped"] or item["escalated"] for item in cleanup):
            outcome["outcome"] = "FAILED"
        (directory / "listeners-final.txt").write_text(
            subprocess.check_output(["ss", "-ltnp"], text=True)
        )
        write_json(directory / "outcome.json", outcome)
    return 0 if outcome["outcome"] == "PASSED" else 1


def status(directory: Path) -> dict:
    """Distinguish absence, active ownership and completed recorded outcome."""
    if not directory.exists():
        return dict(state="absent")
    coordinator = directory / "coordinator.json"
    active = coordinator.exists() and matches(
        json.loads(coordinator.read_text())["identity"]
    )
    services = []
    for path in directory.rglob("*-process.json"):
        record = json.loads(path.read_text())
        services.append(
            dict(
                role=record["role"],
                running=matches(record["child"]),
                guardian_running=matches(record["guardian"]),
                unresolved_group=bool(
                    record["child"] is not None
                    and not matches(record["child"])
                    and group_members(record["child"]["pgid"])
                ),
            )
        )
    outcome = directory / "outcome.json"
    return dict(
        state="running"
        if active
        or any(
            s["running"] or s["guardian_running"] or s["unresolved_group"]
            for s in services
        )
        else "stopped",
        services=services,
        outcome=json.loads(outcome.read_text()) if outcome.exists() else None,
    )


def stop(directory: Path) -> dict:
    """Stop a recorded coordinator and recover only this run's service groups."""
    coordinator = directory / "coordinator.json"
    if coordinator.exists():
        record = json.loads(coordinator.read_text())["identity"]
        if matches(record):
            os.kill(record["pid"], signal.SIGTERM)
            deadline = time.monotonic() + 35
            while matches(record) and time.monotonic() < deadline:
                time.sleep(0.1)
    results = recover(directory / "evidence") + recover(directory / "services")
    return dict(cleanup=results, status=status(directory))


def main() -> None:
    """Dispatch the private node-side commands used by the workstation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "check",
            "coordinate",
            "status",
            "stop",
            "listen",
            "verify",
            "baseline",
        ],
    )
    parser.add_argument(
        "--release", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--host", choices=["node0", "node1"])
    parser.add_argument("--workload", action="store_true")
    parser.add_argument("--address")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.command == "listen":
        with socket.socket() as listener:
            listener.settimeout(15)
            listener.bind((args.address, args.port))
            listener.listen(1)
            print(f"READY {listener.getsockname()[1]}", flush=True)
            with listener.accept()[0] as connection:
                connection.sendall(b"kvc-experiment-network")
        return
    if args.command == "verify":
        result = verify_release(args.release)
    elif args.command == "check":
        result = check(
            args.release, json.loads(args.inventory.read_text()), args.host
        )
    elif args.command in ("coordinate", "baseline"):
        sys.exit(
            coordinate(
                args.release,
                args.directory,
                json.loads(args.inventory.read_text()),
                args.workload,
                baseline=args.command == "baseline",
            )
        )
    elif args.command == "status":
        result = status(args.directory)
    else:
        result = stop(args.directory)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
