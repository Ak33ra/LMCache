# SPDX-License-Identifier: Apache-2.0
"""Deploy and verify the two-node CPU experiment using Python and SSH."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import ipaddress
import json
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile

OUTER_FILES = [
    "experiments/cloudlab/bootstrap.sh",
    "experiments/cloudlab/baseline.sh",
    "experiments/cloudlab/requirements.lock",
    "experiments/cloudlab/node.py",
    "experiments/partitioning/process_host.py",
    "experiments/partitioning/partition_roundtrip.py",
]
SDK_FILES = [
    "experiments/mooncake/build_fenced_sdk.sh",
    "experiments/mooncake/sdk-patches/fenced-tcp.patch",
    "experiments/mooncake/sdk-patches/expected-tree.txt",
]


@dataclass(frozen=True)
class Node:
    """A control-network SSH destination and a distinct cache IPv4 address."""

    ssh: str
    experiment_ip: str


@dataclass(frozen=True)
class Inventory:
    """Validated destinations, experiment addresses and run workspace."""

    node0: Node
    node1: Node
    remote_root: str
    port_base: int

    def document(self) -> dict:
        """Serialize only public deployment settings, never credentials."""
        return dict(
            node0=vars(self.node0),
            node1=vars(self.node1),
            remote_root=self.remote_root,
            port_base=self.port_base,
        )


def load_inventory(path: Path) -> Inventory:
    """Validate all shell/path/network input at the workstation boundary."""
    value = json.loads(path.read_text())
    if set(value) - {"node0", "node1", "remote_root", "port_base"}:
        raise ValueError(
            "Unknown inventory fields; keep credentials in SSH config"
        )
    nodes = []
    for name in ("node0", "node1"):
        item = value[name]
        if set(item) != {"ssh", "experiment_ip"}:
            raise ValueError(f"{name} requires only ssh and experiment_ip")
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", item["ssh"]):
            raise ValueError(f"Invalid SSH destination: {name}")
        address = ipaddress.IPv4Address(item["experiment_ip"])
        if (
            address.is_loopback
            or address.is_multicast
            or address.is_unspecified
        ):
            raise ValueError(f"{name} needs a unicast experiment LAN address")
        nodes.append(Node(item["ssh"], str(address)))
    if nodes[0].experiment_ip == nodes[1].experiment_ip:
        raise ValueError("The two hosts require distinct experiment addresses")
    root = PurePosixPath(value.get("remote_root", "/mydata/kvc"))
    if (
        not root.is_absolute()
        or ".." in root.parts
        or len(root.parts) < 3
        or not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(root))
    ):
        raise ValueError("remote_root must be an absolute workspace path")
    port = value.get("port_base", 24000)
    if type(port) is not int or not 1024 <= port <= 65527:
        raise ValueError("port_base must reserve nine unprivileged TCP ports")
    return Inventory(*nodes, str(root), port)


def checked_id(value: str) -> str:
    """Keep run and release names inside their owned directories."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError(
            "IDs use 1-80 letters, numbers, dots, dashes or underscores"
        )
    return value


class SSH:
    """Workstation SSH options stay local; peer SSH uses node0's own config."""

    def __init__(self, config: Path | None) -> None:
        self.options = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=10",
            "-o",
            "ServerAliveCountMax=3",
        ]
        if config:
            self.options += ["-F", str(config.resolve())]

    def argv(self, node: Node, command: str) -> list[str]:
        """Construct an SSH command without host-key or forwarding overrides."""
        return [*self.options, node.ssh, command]

    def run(
        self, node: Node, command: str, **kwargs
    ) -> subprocess.CompletedProcess:
        """Run a command with explicit transport failure propagation."""
        return subprocess.run(self.argv(node, command), check=True, **kwargs)


def package(project: Path, engine: Path, destination: Path) -> tuple[str, Path]:
    """Bundle clean tracked engine source and an explicit outer-file allowlist."""
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(engine),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ]
    )
    if dirty:
        raise RuntimeError("LMCache tracked source must be clean before setup")
    head = subprocess.check_output(
        ["git", "-C", str(engine), "rev-parse", "HEAD"], text=True
    ).strip()
    bundle = destination / "LMCache.bundle"
    subprocess.run(
        [
            "git",
            "-C",
            str(engine),
            "-c",
            "pack.threads=1",
            "bundle",
            "create",
            str(bundle),
            "HEAD",
            "--tags",
        ],
        check=True,
    )
    writer = Path(__file__).resolve().parents[2]
    sources = {"LMCache.bundle": bundle}
    sources.update({name: writer / name for name in OUTER_FILES})
    sources.update({name: project / name for name in SDK_FILES})
    manifest = dict(
        lmcache_head=head,
        files={
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sources.items()
        },
    )
    payload = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    release_id = hashlib.sha256(payload.encode()).hexdigest()[:20]
    manifest_path = destination / "source-manifest.json"
    manifest_path.write_text(payload)
    sources["source-manifest.json"] = manifest_path
    archive = destination / f"{release_id}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, path in sources.items():
            tar.add(path, arcname=name, recursive=False)
    return release_id, archive


def setup(
    inventory: Inventory, ssh: SSH, args: argparse.Namespace, output: Path
) -> None:
    """Build the exact packaged release on both hosts, with separate logs."""
    with tempfile.TemporaryDirectory(prefix="kvc-source-") as temporary:
        release_id, archive = package(
            args.project_root.resolve(),
            args.lmcache_root.resolve(),
            Path(temporary),
        )
        release = f"{inventory.remote_root}/releases/{release_id}"
        (output / "release.txt").write_text(release_id + "\n")
        shutil.copy2(Path(temporary) / "source-manifest.json", output)

        def install(name: str) -> None:
            node = getattr(inventory, name)
            with (output / f"{name}-setup.log").open("wb") as log:
                workspace = shlex.quote(inventory.remote_root)
                prepare = (
                    f"if [ ! -d {workspace} ]; then "
                    f"mkdir -p {workspace} 2>/dev/null || "
                    'sudo -n install -d -m 0750 -o "$(id -u)" -g "$(id -g)" '
                    f"{workspace}; fi; test -w {workspace}"
                )
                ssh.run(node, prepare, stdout=log, stderr=subprocess.STDOUT)
                with archive.open("rb") as source:
                    ssh.run(
                        node,
                        f"mkdir -p {shlex.quote(release)} && "
                        f"tar -xz -C {shlex.quote(release)}",
                        stdin=source,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                bootstrap = shlex.quote(
                    f"{release}/experiments/cloudlab/bootstrap.sh"
                )
                verify = shlex.join(
                    [
                        "python3",
                        f"{release}/experiments/cloudlab/node.py",
                        "verify",
                    ]
                )
                ssh.run(
                    node,
                    f"if ! {verify}; then bash {bootstrap}; fi",
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                ssh.run(node, verify, stdout=log, stderr=subprocess.STDOUT)
            print(f"{name}: release {release_id} ready", flush=True)

        for name in ("node0", "node1"):
            install(name)
        for node in (inventory.node0, inventory.node1):
            ssh.run(
                node,
                f"echo {shlex.quote(release_id)} > "
                f"{shlex.quote(inventory.remote_root + '/current-release')}",
            )
        (output / "outcome.json").write_text(
            json.dumps(
                dict(
                    outcome="PASSED",
                    release=release_id,
                    scope="fresh CPU setup",
                )
            )
            + "\n"
        )


def select_release(inventory: Inventory, ssh: SSH) -> str:
    """Require both hosts to select the same completed release."""
    releases = [
        ssh.run(
            node,
            "cat " + shlex.quote(inventory.remote_root + "/current-release"),
            capture_output=True,
            text=True,
        ).stdout.strip()
        for node in (inventory.node0, inventory.node1)
    ]
    if releases[0] != releases[1]:
        raise RuntimeError("Hosts select different releases; rerun setup")
    return checked_id(releases[0])


def node_command(inventory: Inventory, run_id: str, operation: str) -> str:
    """Find a run's original helper even if a later setup selects another release."""
    directory = f"{inventory.remote_root}/runs/{checked_id(run_id)}"
    script = (
        f"if [ ! -f {shlex.quote(directory + '/release.txt')} ]; then "
        'printf \'{"state":"absent"}\\n\'; exit 0; fi; '
        f"release=$(cat {shlex.quote(directory + '/release.txt')}); "
        f"exec python3 {shlex.quote(inventory.remote_root)}/releases/"
        f'"$release"/experiments/cloudlab/node.py {operation} '
        f"--directory {shlex.quote(directory)}"
    )
    return script


def stop(inventory: Inventory, ssh: SSH, run_id: str) -> dict:
    """Attempt both hosts even when one is unreachable; preserve that distinction."""
    result = {}
    for name in ("node0", "node1"):
        try:
            response = ssh.run(
                getattr(inventory, name),
                node_command(inventory, run_id, "stop"),
                capture_output=True,
                text=True,
            )
            result[name] = json.loads(response.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
            result[name] = dict(
                state="unreachable"
                if getattr(error, "returncode", 0) == 255
                else "error",
                error=str(error),
            )
    return result


def collect(inventory: Inventory, ssh: SSH, run_id: str, output: Path) -> dict:
    """Collect complete host evidence without modifying remote result directories."""
    result = {}
    for name in ("node0", "node1"):
        path = output / f"{name}.tar.gz"
        try:
            with path.open("xb") as archive:
                ssh.run(
                    getattr(inventory, name),
                    shlex.join(
                        [
                            "tar",
                            "-czf",
                            "-",
                            "-C",
                            inventory.remote_root + "/runs",
                            run_id,
                        ]
                    ),
                    stdout=archive,
                    stderr=subprocess.PIPE,
                )
            with tarfile.open(path) as archive:
                target = output / name
                target.mkdir()
                for member in archive.getmembers():
                    relative = PurePosixPath(member.name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise RuntimeError("Unsafe collected artifact path")
                    if not (member.isfile() or member.isdir()):
                        raise RuntimeError(
                            "Collection accepts regular evidence files only"
                        )
                archive.extractall(target, filter="data")
            result[name] = dict(collected=True)
        except (
            OSError,
            subprocess.CalledProcessError,
            tarfile.TarError,
            RuntimeError,
        ) as error:
            result[name] = dict(collected=False, error=str(error))
    return result


def run(
    inventory: Inventory, ssh: SSH, args: argparse.Namespace, output: Path
) -> None:
    """Reserve fresh host directories, run on node0, clean up, then collect."""
    baseline = args.command == "baseline"
    job_hosts = ("node0", "node1") if baseline else ("node0",)
    run_id = checked_id(args.run_id)
    release_id = select_release(inventory, ssh)
    release = f"{inventory.remote_root}/releases/{release_id}"
    directory = f"{inventory.remote_root}/runs/{run_id}"
    outcome = dict(
        outcome="BLOCKED",
        run_id=run_id,
        release=release_id,
        workload="REQUESTED" if args.workload else "SKIPPED_BY_REQUEST",
    )
    reserved = []
    try:
        for name in ("node0", "node1"):
            node = getattr(inventory, name)
            ssh.run(
                node,
                f"mkdir -p {shlex.quote(inventory.remote_root + '/runs')} && "
                f"mkdir {shlex.quote(directory)}",
            )
            reserved.append(name)
            metadata = dict(
                inventory=inventory.document(),
                release=release_id,
                run_id=run_id,
                workload=args.workload,
            )
            with tempfile.TemporaryDirectory(prefix="kvc-run-") as temporary:
                path = Path(temporary)
                (path / "inventory.json").write_text(
                    json.dumps(inventory.document())
                )
                (path / "release.txt").write_text(release_id + "\n")
                (path / "run.json").write_text(
                    json.dumps(metadata, indent=2) + "\n"
                )
                archive_path = path / "run.tar"
                with tarfile.open(archive_path, "w") as archive:
                    for filename in (
                        "inventory.json",
                        "release.txt",
                        "run.json",
                    ):
                        archive.add(path / filename, arcname=filename)
                with archive_path.open("rb") as archive:
                    ssh.run(
                        node,
                        shlex.join(["tar", "-xf", "-", "-C", directory]),
                        stdin=archive,
                    )
            ssh.run(
                node,
                shlex.join(
                    [
                        "cp",
                        release + "/source-manifest.json",
                        release + "/build-manifest.json",
                        release + "/packages.txt",
                        release + "/system-packages.txt",
                        directory,
                    ]
                ),
            )
            check = [
                f"{release}/venv/bin/python",
                f"{release}/experiments/cloudlab/node.py",
                "verify" if baseline else "check",
                "--inventory",
                directory + "/inventory.json",
                "--host",
                name,
            ]
            response = ssh.run(node, shlex.join(check), capture_output=True)
            (output / f"{name}-preflight.json").write_bytes(response.stdout)
            ssh.run(
                node,
                "cat > " + shlex.quote(directory + "/preflight.json"),
                input=response.stdout,
            )
        outcome["outcome"] = "FAILED"
        outcome["job_exits"] = {}
        for name in job_hosts:
            command = [
                f"{release}/venv/bin/python",
                f"{release}/experiments/cloudlab/node.py",
                "baseline" if baseline else "coordinate",
                "--directory",
                directory,
                "--inventory",
                directory + "/inventory.json",
            ]
            if args.workload:
                command.append("--workload")
            with (output / f"{name}-coordinator.log").open("wb") as log:
                process = subprocess.Popen(
                    ssh.argv(
                        getattr(inventory, name), "exec " + shlex.join(command)
                    ),
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    outcome["job_exits"][name] = process.wait()
                finally:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=40)
    except (Exception, KeyboardInterrupt) as error:
        outcome["error"] = str(error)
    finally:
        cleanup = {}
        for name in reserved:
            try:
                result = ssh.run(
                    getattr(inventory, name),
                    node_command(inventory, run_id, "stop"),
                    capture_output=True,
                    text=True,
                )
                cleanup[name] = json.loads(result.stdout)
            except Exception as error:
                cleanup[name] = dict(error=str(error))
        outcome["cleanup"] = cleanup
        outcome["collection"] = collect(inventory, ssh, run_id, output)
        complete = (
            len(reserved) == 2
            and len(outcome.get("job_exits", {})) == len(job_hosts)
            and all(code == 0 for code in outcome.get("job_exits", {}).values())
            and all(
                item.get("collected") for item in outcome["collection"].values()
            )
        )
        if complete:
            remotes = [
                output / name / run_id / "outcome.json" for name in job_hosts
            ]
            complete = (
                all(
                    remote.exists()
                    and json.loads(remote.read_text())["outcome"] == "PASSED"
                    for remote in remotes
                )
                and all(
                    item.get("status", {}).get("state") == "stopped"
                    for item in cleanup.values()
                )
                and all(
                    not entry.get("escalated") and entry.get("stopped")
                    for item in cleanup.values()
                    for entry in item.get("cleanup", [])
                )
            )
        if complete:
            outcome["outcome"] = "PASSED"
        (output / "outcome.json").write_text(
            json.dumps(outcome, indent=2) + "\n"
        )
    if outcome["outcome"] != "PASSED":
        raise RuntimeError(
            f"{outcome['outcome']}; inspect {output}/outcome.json"
        )


def main() -> None:
    """Expose the small operator CLI; every output directory is fresh."""
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--ssh-config", type=Path)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--lmcache-root", type=Path, default=root / "LMCache")
    parser.add_argument(
        "command",
        choices=[
            "inventory",
            "setup",
            "baseline",
            "run",
            "status",
            "stop",
            "collect",
        ],
    )
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workload", action="store_true")
    args = parser.parse_args()
    if args.command == "inventory":
        print(
            (Path(__file__).parent / "inventory.example.json").read_text(),
            end="",
        )
        return
    if args.inventory is None:
        parser.error("--inventory is required")
    inventory = load_inventory(args.inventory)
    ssh = SSH(args.ssh_config)
    if (
        args.command in ("run", "status", "stop", "collect", "baseline")
        and not args.run_id
    ):
        parser.error("--run-id is required")
    if args.run_id:
        checked_id(args.run_id)
    if args.command == "status":
        results = {}
        for name in ("node0", "node1"):
            try:
                response = ssh.run(
                    getattr(inventory, name),
                    node_command(inventory, args.run_id, "status"),
                    capture_output=True,
                    text=True,
                )
                results[name] = json.loads(response.stdout)
            except (
                subprocess.CalledProcessError,
                json.JSONDecodeError,
            ) as error:
                results[name] = dict(
                    state="unreachable"
                    if getattr(error, "returncode", 0) == 255
                    else "error",
                    error=str(error),
                )
        print(json.dumps(results, indent=2))
        if any(
            value["state"] in ("unreachable", "error")
            for value in results.values()
        ):
            sys.exit(2)
        return
    if args.command == "stop":
        results = stop(inventory, ssh, args.run_id)
        print(json.dumps(results, indent=2))
        if any(
            value.get("state") in ("unreachable", "error")
            or value.get("status", {}).get("state") == "running"
            for value in results.values()
        ):
            sys.exit(1)
        return
    output = args.output_dir or root / "experiments/cloudlab/results" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + args.command
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f"Evidence directory: {output}", flush=True)
    try:
        if args.command == "setup":
            setup(inventory, ssh, args, output)
        elif args.command in ("run", "baseline"):
            run(inventory, ssh, args, output)
        else:
            result = collect(inventory, ssh, args.run_id, output)
            print(json.dumps(result, indent=2))
            if not all(r["collected"] for r in result.values()):
                sys.exit(1)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        outcome = output / "outcome.json"
        if not outcome.exists():
            outcome.write_text(
                json.dumps(
                    dict(
                        outcome="BLOCKED",
                        operation=args.command,
                        error=str(error),
                    ),
                    indent=2,
                )
                + "\n"
            )
        raise


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
