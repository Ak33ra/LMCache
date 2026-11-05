# SPDX-License-Identifier: Apache-2.0
"""Own experiment service groups and read process maps on their actual host."""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import argparse
import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time


def write_json(path: Path, value: dict) -> None:
    """Publish a complete ledger record atomically."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def identity(pid: int) -> dict | None:
    """Read a Linux process identity; zombies count as exited."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    if fields[0] == "Z":
        return None
    return dict(
        pid=pid,
        pgid=int(fields[2]),
        start_ticks=int(fields[19]),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    )


def matches(record: dict | None) -> bool:
    """Require boot, PID, process group and start time to still match."""
    return record is not None and identity(record["pid"]) == record


def group_members(pgid: int) -> list[dict]:
    """Observe live members of one recorded process group."""
    members = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            member = identity(int(path.name))
            if member is not None and member["pgid"] == pgid:
                members.append(member)
    return members


def terminate(record: dict, timeout: float = 20) -> bool:
    """Stop the owned group, including descendants whose leader exits first."""
    if not matches(record):
        if (
            record is not None
            and identity(record["pid"]) is None
            and group_members(record["pgid"])
        ):
            raise RuntimeError("Group survives without a verifiable owner")
        return False
    owned = group_members(record["pgid"])
    if not matches(record):
        raise RuntimeError("Process identity changed during cleanup")
    try:
        os.killpg(record["pgid"], signal.SIGTERM)
    except ProcessLookupError:
        return False
    deadline = time.monotonic() + timeout
    while group_members(record["pgid"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    if not group_members(record["pgid"]):
        return False
    if not any(matches(member) for member in owned):
        raise RuntimeError("Cannot prove ownership of remaining group members")
    os.killpg(record["pgid"], signal.SIGKILL)
    deadline = time.monotonic() + 5
    while group_members(record["pgid"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    if group_members(record["pgid"]):
        raise RuntimeError("Owned group remains after SIGKILL")
    return True


def recover(directory: Path) -> list[dict]:
    """Recover service guardians after lost control, without signaling reused PIDs."""
    results = []
    for path in sorted(directory.glob("*-process.json")):
        record = json.loads(path.read_text())
        guardian = record["guardian"]
        if matches(guardian):
            os.kill(guardian["pid"], signal.SIGTERM)
            deadline = time.monotonic() + 30
            while matches(guardian) and time.monotonic() < deadline:
                time.sleep(0.1)
        if matches(guardian):
            results.append(
                dict(
                    role=record["role"],
                    stopped=False,
                    error="guardian did not finish cleanup",
                )
            )
            continue
        record = json.loads(path.read_text())
        try:
            escalated = terminate(record["child"])
        except RuntimeError as error:
            results.append(
                dict(
                    role=record["role"],
                    stopped=False,
                    escalated=False,
                    error=str(error),
                )
            )
            continue
        results.append(
            dict(
                role=record["role"],
                stopped=not matches(record["child"]),
                escalated=escalated or record.get("escalated", False),
            )
        )
    return results


class Process:
    """A service whose guardian owns cleanup and host-local observations."""

    def __init__(self, control: subprocess.Popen, record: dict) -> None:
        self.control = control
        self.pid = record["child"]["pid"]
        self.returncode = None

    def request(self, operation: str) -> dict:
        """Exchange one command with the service's host-local guardian."""
        self.control.stdin.write(json.dumps({"operation": operation}) + "\n")
        self.control.stdin.flush()
        line = self.control.stdout.readline()
        if not line:
            raise RuntimeError(
                "Service guardian disconnected; run stop to recover"
            )
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def poll(self) -> int | None:
        """Return the child exit code from its actual parent."""
        self.returncode = self.request("poll")["returncode"]
        return self.returncode

    def read_maps(self) -> str:
        """Read maps from the service host, never the coordinator's /proc."""
        return self.request("maps")["maps"]


class ProcessHost:
    """Select local or peer SSH service execution with identical ownership."""

    def __init__(
        self,
        source: Path,
        output: Path,
        address: str = "127.0.0.1",
        ssh: str | None = None,
        helper: Path | None = None,
        python: Path | None = None,
    ) -> None:
        self.source = source
        self.output = output
        self.address = address
        self.ssh = ssh
        self.helper = helper or Path(__file__).resolve()
        self.python = python or Path(sys.executable)

    @contextmanager
    def service(
        self, name: str, command: list[str], local_output: Path
    ) -> Iterator[Process]:
        """Start a service; require clean graceful shutdown before returning."""
        environment = dict(
            CUDA_VISIBLE_DEVICES="",
            LMCACHE_TRACK_USAGE="false",
            MC_TCP_BIND_ADDRESS=self.address,
            MC_NUM_TCP_IO_THREADS="2",
            MC_TCP_REQUIRE_GENERATION="1",
            MC_NOF_BACKEND="none",
            LMCACHE_COORDINATOR_URL="",
            PYTHONPATH=os.environ.get("PYTHONPATH", ""),
        )
        spec = dict(
            role=name,
            command=command,
            environment=environment,
            cwd=str(self.source),
            output=str(self.output),
        )
        argv = [str(self.python), str(self.helper), "serve"]
        if self.ssh:
            argv = [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "ServerAliveInterval=10",
                "-o",
                "ServerAliveCountMax=3",
                self.ssh,
                "exec " + shlex.join(argv),
            ]
        with (local_output / f"{name}-control.log").open("w") as control_log:
            control = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=control_log,
                text=True,
                start_new_session=True,
            )
            process = None
            try:
                control.stdin.write(json.dumps(spec) + "\n")
                control.stdin.flush()
                line = control.stdout.readline()
                if not line:
                    raise RuntimeError(f"{name} guardian could not start")
                record = json.loads(line)
                if "error" in record:
                    raise RuntimeError(record["error"])
                process = Process(control, record)
                (local_output / f"{name}-pid.txt").write_text(
                    f"{process.pid}\n"
                )
                (local_output / f"{name}-command.json").write_text(
                    json.dumps(command) + "\n"
                )
                yield process
            finally:
                try:
                    if process is not None:
                        stopped = process.request("stop")
                        (local_output / f"{name}.log").write_text(
                            stopped["log"]
                        )
                        (local_output / f"{name}-exit.txt").write_text(
                            str(stopped["returncode"]) + "\n"
                        )
                        write_json(
                            local_output / f"{name}-cleanup.json", stopped
                        )
                        if stopped["escalated"]:
                            raise RuntimeError(f"{name} required SIGKILL")
                        if name in ("a", "b") and not stopped["graceful"]:
                            raise RuntimeError(
                                f"{name} did not shut down gracefully"
                            )
                        if stopped["returncode"] not in (0, -signal.SIGTERM):
                            raise RuntimeError(
                                f"{name} exited with {stopped['returncode']}"
                            )
                finally:
                    try:
                        control.stdin.close()
                    except BrokenPipeError:
                        pass
                    try:
                        control.wait(timeout=35)
                    except subprocess.TimeoutExpired:
                        control.terminate()
                        control.wait(timeout=5)
                    control.stdout.close()


def serve() -> None:
    """Serve one owned child until stop, controller EOF, or SIGTERM."""
    spec = json.loads(sys.stdin.readline())
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    name = spec["role"]
    ledger = output / f"{name}-process.json"
    if ledger.exists():
        raise FileExistsError(ledger)
    environment = dict(os.environ, **spec["environment"])
    for key in ("LD_LIBRARY_PATH", "MC_STORE_LOCAL_HOT_CACHE_SIZE"):
        environment.pop(key, None)
    child = None
    record = dict(
        run_id=output.parent.name,
        host=socket.gethostname(),
        role=name,
        guardian=identity(os.getpid()),
        child=None,
        spec_sha256=hashlib.sha256(
            json.dumps(spec, sort_keys=True).encode()
        ).hexdigest(),
    )

    def interrupted(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    def reply(value: dict) -> None:
        print(json.dumps(value), flush=True)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    log_path = output / f"{name}.log"
    with log_path.open("xb") as log:
        try:
            child = subprocess.Popen(
                spec["command"],
                cwd=spec["cwd"],
                env=environment,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            record["child"] = identity(child.pid)
            if record["child"] is None:
                raise RuntimeError(f"{name} exited during launch")
            write_json(ledger, record)
            reply(record)
            for line in sys.stdin:
                operation = json.loads(line)["operation"]
                if operation == "stop":
                    break
                if operation == "maps":
                    if not matches(record["child"]):
                        reply({"error": f"{name} is no longer running"})
                    else:
                        (output / f"{name}-listeners.txt").write_text(
                            subprocess.check_output(["ss", "-ltnp"], text=True)
                        )
                        reply(
                            {
                                "maps": Path(
                                    f"/proc/{child.pid}/maps"
                                ).read_text()
                            }
                        )
                elif operation == "poll":
                    reply({"returncode": child.poll()})
                else:
                    raise ValueError(f"Unknown operation: {operation}")
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            if child is not None:
                escalated = False
                if record["child"] is not None:
                    escalated = terminate(record["child"])
                child.wait(timeout=5)
                log.flush()
                text = log_path.read_text(errors="replace")
                record.update(
                    returncode=child.returncode,
                    escalated=escalated,
                    graceful="Application shutdown complete." in text,
                )
                write_json(ledger, record)
                try:
                    reply(
                        dict(
                            returncode=child.returncode,
                            escalated=escalated,
                            graceful=record["graceful"],
                            log=text,
                        )
                    )
                except BrokenPipeError:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["serve", "stop"])
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    try:
        if args.operation == "serve":
            serve()
        else:
            print(json.dumps(recover(args.directory)))
    except KeyboardInterrupt:
        sys.exit(130)
