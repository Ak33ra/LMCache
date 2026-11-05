# SPDX-License-Identifier: Apache-2.0
"""Exercise installed host helpers through real SSH; no Torch on the caller."""

from pathlib import Path
import argparse
import json
import shlex
import subprocess
import sys

from cloudlab import SSH, checked_id, collect, load_inventory, node_command
from cloudlab import select_release, stop

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "partitioning"))
from process_host import Process


def main() -> None:
    """Check peer maps, partial startup, lost control, recovery and collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--ssh-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--scope",
        required=True,
        choices=["one-host-container-emulation", "two-physical-hosts"],
    )
    args = parser.parse_args()
    run_id = checked_id(args.run_id)
    inventory = load_inventory(args.inventory)
    ssh = SSH(args.ssh_config)
    release_id = select_release(inventory, ssh)
    release = f"{inventory.remote_root}/releases/{release_id}"
    directory = f"{inventory.remote_root}/runs/{run_id}"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = dict(outcome="FAILED", scope=args.scope, run_id=run_id, checks=[])
    controls = []
    reserved = []
    try:
        for name in ("node0", "node1"):
            node = getattr(inventory, name)
            ssh.run(
                node,
                f"mkdir {shlex.quote(directory)} && "
                f"echo {shlex.quote(release_id)} > "
                f"{shlex.quote(directory + '/release.txt')}",
            )
            reserved.append(name)
            helper = [
                f"{release}/venv/bin/python",
                f"{release}/experiments/partitioning/process_host.py",
                "serve",
            ]
            control = subprocess.Popen(
                ssh.argv(node, "exec " + shlex.join(helper)),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            controls.append(control)
            spec = dict(
                role="master",
                command=[
                    f"{release}/venv/bin/python",
                    "-c",
                    "import socket,time; print(socket.gethostname(),flush=True); "
                    "time.sleep(120)",
                ],
                cwd=release,
                environment={},
                output=directory + "/services",
            )
            control.stdin.write(json.dumps(spec) + "\n")
            control.stdin.flush()
            record = json.loads(control.stdout.readline())
            handle = Process(control, record)
            maps = handle.read_maps()
            if "python" not in maps or handle.poll() is not None:
                raise RuntimeError(f"{name}: remote service/maps check failed")
            (args.output_dir / f"{name}-maps.txt").write_text(maps)
            (args.output_dir / f"{name}-identity.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
        result["checks"].append("both SSH-owned services and host-local maps")
        controls[0].stdin.close()
        if controls[0].wait(timeout=30):
            raise RuntimeError("EOF cleanup guardian failed")
        state = json.loads(
            ssh.run(
                inventory.node0,
                node_command(inventory, run_id, "status"),
                capture_output=True,
                text=True,
            ).stdout
        )
        if state["state"] != "stopped":
            raise RuntimeError("Lost control left node0 service running")
        result["checks"].append("controller EOF cleans owned service")
        controls[1].kill()
        controls[1].wait(timeout=10)
        # SIGKILL of local SSH can deliver either EOF or HUP remotely.
        cleanup = stop(inventory, ssh, run_id)
        if any(
            item.get("status", {}).get("state") != "stopped"
            for item in cleanup.values()
        ):
            raise RuntimeError(f"Interrupted SSH cleanup failed: {cleanup}")
        result["checks"].append("interrupted SSH and recorded recovery")
        again = stop(inventory, ssh, run_id)
        if any(
            item.get("status", {}).get("state") != "stopped"
            for item in again.values()
        ):
            raise RuntimeError("Repeated stop did not remain stopped")
        result["checks"].append("repeated stop")
        helper = [
            f"{release}/venv/bin/python",
            f"{release}/experiments/partitioning/process_host.py",
            "serve",
        ]
        bad = dict(
            role="failed",
            command=["/missing/kvc-executable"],
            cwd=release,
            environment={},
            output=directory + "/services",
        )
        partial = subprocess.run(
            ssh.argv(inventory.node1, "exec " + shlex.join(helper)),
            input=json.dumps(bad) + "\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        if partial.returncode == 0:
            raise RuntimeError("Invalid service unexpectedly started")
        (args.output_dir / "partial-start.log").write_text(partial.stderr)
        result["checks"].append("partial service launch returns failure")
        collected = collect(inventory, ssh, run_id, args.output_dir)
        if not all(item["collected"] for item in collected.values()):
            raise RuntimeError(f"Failure collection failed: {collected}")
        for name in ("node0", "node1"):
            ledger = json.loads(
                (
                    args.output_dir
                    / name
                    / run_id
                    / "services/master-process.json"
                ).read_text()
            )
            if ledger["escalated"] or "returncode" not in ledger:
                raise RuntimeError(
                    f"{name}: incomplete graceful cleanup ledger"
                )
        result["checks"].append(
            "logs and exit ledgers collect after partial failure"
        )
        result["outcome"] = "PASSED"
    finally:
        for control in controls:
            if not control.stdin.closed:
                try:
                    control.stdin.close()
                except BrokenPipeError:
                    pass
            if control.poll() is None:
                control.terminate()
                control.wait(timeout=30)
            control.stdout.close()
            control.stderr.close()
        for name in reserved:
            ssh.run(
                getattr(inventory, name),
                node_command(inventory, run_id, "stop"),
                stdout=subprocess.DEVNULL,
            )
        (args.output_dir / "outcome.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
