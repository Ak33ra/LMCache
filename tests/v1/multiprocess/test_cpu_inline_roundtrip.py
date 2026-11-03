# SPDX-License-Identifier: Apache-2.0
"""Process ownership checks for the CPU inline verification example."""

# Standard
from typing import IO
import subprocess

# Third Party
import pytest

# First Party
from examples.multi_process import cpu_inline_roundtrip


@pytest.mark.skipif(
    cpu_inline_roundtrip.torch_device_type != "cpu", reason="CPU verification example"
)
def test_client_construction_failure_reaps_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed client factory must not leave its real server process running."""
    children: list[subprocess.Popen[bytes]] = []
    original_popen = subprocess.Popen

    def spawn(
        command: list[str], *, env: dict[str, str], stdout: IO[str], stderr: int
    ) -> subprocess.Popen[bytes]:
        child = original_popen(command, env=env, stdout=stdout, stderr=stderr)
        children.append(child)
        return child

    def fail_client(url: str) -> None:
        assert url.startswith("tcp://127.0.0.1:")
        assert len(children) == 1 and children[0].poll() is None
        raise RuntimeError("injected client construction failure")

    monkeypatch.setattr(cpu_inline_roundtrip.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        cpu_inline_roundtrip.RequestClientFactory, "create", fail_client
    )
    try:
        with pytest.raises(RuntimeError, match="injected client construction failure"):
            cpu_inline_roundtrip.main()
        assert len(children) == 1
        assert children[0].poll() is not None
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
