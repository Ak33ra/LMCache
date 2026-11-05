# SPDX-License-Identifier: Apache-2.0
"""Exercise real Linux child ownership without Torch, mocks or service skips."""

from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "partitioning"))
from process_host import ProcessHost, identity, matches, recover, write_json

HELPER = Path(__file__).resolve().parents[1] / "partitioning/process_host.py"


def await_exit(record: dict) -> None:
    """Require the actual recorded process to disappear before the deadline."""
    deadline = time.monotonic() + 10
    while matches(record) and time.monotonic() < deadline:
        time.sleep(0.02)
    if matches(record):
        raise AssertionError(f"Owned process survived: {record}")


class LifecycleTest(unittest.TestCase):
    """Check graceful ownership, lost control, partial startup and PID reuse."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="kvc-lifecycle-")
        self.output = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def launch_guardian(self) -> tuple[subprocess.Popen, dict]:
        """Launch a real service through the same newline-JSON entrypoint as SSH."""
        process = subprocess.Popen(
            [sys.executable, str(HELPER), "serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        spec = dict(
            role="master",
            command=[sys.executable, "-c", "import time; time.sleep(120)"],
            environment={},
            cwd=str(self.output),
            output=str(self.output),
        )
        process.stdin.write(json.dumps(spec) + "\n")
        process.stdin.flush()
        record = json.loads(process.stdout.readline())
        self.addCleanup(process.stdout.close)
        self.addCleanup(process.stdin.close)
        self.addCleanup(lambda: recover(self.output))
        return process, record

    def test_service_reads_child_maps_and_cleans_group(self) -> None:
        host = ProcessHost(self.output, self.output)
        with host.service(
            "master",
            [sys.executable, "-c", "import time; time.sleep(120)"],
            self.output,
        ) as child:
            process_identity = identity(child.pid)
            self.assertIsNone(child.poll())
            self.assertIn("python", child.read_maps())
            self.assertNotEqual(child.pid, os.getpid())
        self.assertFalse(matches(process_identity))
        self.assertEqual(
            recover(self.output),
            [dict(role="master", stopped=True, escalated=False)],
        )
        self.assertEqual(
            recover(self.output),
            [dict(role="master", stopped=True, escalated=False)],
        )

    def test_completed_baseline_job_needs_no_http_shutdown_marker(self) -> None:
        host = ProcessHost(self.output, self.output)
        with host.service(
            "baseline", [sys.executable, "-c", "print('done')"], self.output
        ) as child:
            deadline = time.monotonic() + 10
            while child.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(child.returncode, 0)
        self.assertEqual((self.output / "baseline-exit.txt").read_text(), "0\n")

    def test_controller_eof_stops_service(self) -> None:
        guardian, record = self.launch_guardian()
        guardian.stdin.close()
        guardian.wait(timeout=10)
        await_exit(record["child"])
        self.assertFalse(
            json.loads((self.output / "master-process.json").read_text())[
                "escalated"
            ]
        )

    def test_ssh_hangup_stops_service(self) -> None:
        guardian, record = self.launch_guardian()
        os.kill(guardian.pid, signal.SIGHUP)
        guardian.wait(timeout=10)
        await_exit(record["child"])

    def test_recovery_after_guardian_is_killed(self) -> None:
        guardian, record = self.launch_guardian()
        guardian.kill()
        guardian.wait(timeout=10)
        self.assertTrue(matches(record["child"]))
        result = recover(self.output)
        self.assertTrue(result[0]["stopped"])
        await_exit(record["child"])

    def test_stale_identity_does_not_signal_unrelated_process(self) -> None:
        with subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            start_new_session=True,
        ) as unrelated:
            try:
                stale = dict(identity(unrelated.pid))
                stale["start_ticks"] += 1
                write_json(
                    self.output / "master-process.json",
                    dict(role="master", guardian=None, child=stale),
                )
                recover(self.output)
                self.assertIsNone(unrelated.poll())
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=10)

    def test_group_member_cannot_outlive_service_leader(self) -> None:
        host = ProcessHost(self.output, self.output)
        child_pid_file = self.output / "descendant.pid"
        descendant = (
            "import os,signal,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"Path({str(child_pid_file)!r}).write_text(str(os.getpid())); "
            "time.sleep(120)"
        )
        command = [
            sys.executable,
            "-c",
            "import subprocess,sys,time; "
            f"subprocess.Popen([sys.executable, '-c', {descendant!r}]); "
            "time.sleep(120)",
        ]
        descendant_identity = None
        try:
            with self.assertRaisesRegex(RuntimeError, "SIGKILL"):
                with host.service("master", command, self.output):
                    deadline = time.monotonic() + 10
                    while (
                        not child_pid_file.exists()
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.02)
                    descendant_identity = identity(
                        int(child_pid_file.read_text())
                    )
            await_exit(descendant_identity)
        finally:
            if matches(descendant_identity):
                os.kill(descendant_identity["pid"], signal.SIGKILL)

    def test_partial_start_does_not_leave_a_guardian(self) -> None:
        host = ProcessHost(self.output, self.output)
        with self.assertRaises(RuntimeError):
            with host.service(
                "master", ["/definitely/missing/kvc-command"], self.output
            ):
                self.fail("Missing command unexpectedly launched")
        self.assertFalse(list(self.output.glob("*-process.json")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
