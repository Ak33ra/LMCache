# SPDX-License-Identifier: Apache-2.0
"""Check public inventory and exact-source packaging boundaries without SSH."""

from pathlib import Path
import hashlib
import json
import subprocess
import tarfile
import tempfile
import unittest

from cloudlab import SDK_FILES, load_inventory, package


class PackagingTest(unittest.TestCase):
    """Exercise real Git source bundles and rejection of undeclared input."""

    def test_bundle_is_exact_repeatable_and_excludes_untracked_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="kvc-package-") as temporary:
            root = Path(temporary)
            engine = root / "engine"
            engine.mkdir()
            subprocess.run(["git", "init", "-q", str(engine)], check=True)
            (engine / "source.py").write_text('print("exact source")\n')
            for index in range(256):
                (engine / f"payload-{index:03}.txt").write_text(
                    "shared text for delta compression\n" * 128 + str(index)
                )
            subprocess.run(["git", "-C", str(engine), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(engine),
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@localhost",
                    "commit",
                    "-qm",
                    "Source fixture",
                ],
                check=True,
            )
            (engine / "private-key").write_text("DO NOT PACKAGE THIS")
            project = root / "project"
            for name in SDK_FILES:
                path = project / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture SDK input\n")
            staging = root / "staging"
            staging.mkdir()
            subprocess.run(
                ["git", "-C", str(engine), "config", "pack.threads", "1"],
                check=True,
            )
            release_id, archive_path = package(project, engine, staging)
            repeated = root / "repeated"
            repeated.mkdir()
            subprocess.run(
                ["git", "-C", str(engine), "config", "pack.threads", "4"],
                check=True,
            )
            repeated_id, _ = package(project, engine, repeated)
            self.assertEqual(release_id, repeated_id)
            self.assertEqual(
                (staging / "LMCache.bundle").read_bytes(),
                (repeated / "LMCache.bundle").read_bytes(),
            )
            with tarfile.open(archive_path) as archive:
                archive.extractall(root / "unpacked", filter="data")
            unpacked = root / "unpacked"
            manifest = json.loads(
                (unpacked / "source-manifest.json").read_text()
            )
            self.assertEqual(
                release_id,
                hashlib.sha256(
                    (unpacked / "source-manifest.json").read_bytes()
                ).hexdigest()[:20],
            )
            for name, digest in manifest["files"].items():
                self.assertEqual(
                    hashlib.sha256((unpacked / name).read_bytes()).hexdigest(),
                    digest,
                )
            clone = root / "clone"
            subprocess.run(
                [
                    "git",
                    "clone",
                    "-q",
                    str(unpacked / "LMCache.bundle"),
                    str(clone),
                ],
                check=True,
            )
            self.assertEqual(
                (clone / "source.py").read_text(),
                (engine / "source.py").read_text(),
            )
            self.assertFalse((clone / "private-key").exists())
            head = subprocess.check_output(
                ["git", "-C", str(clone), "rev-parse", "HEAD"], text=True
            ).strip()
            self.assertEqual(head, manifest["lmcache_head"])
            (engine / "source.py").write_text("dirty\n")
            with self.assertRaisesRegex(RuntimeError, "clean"):
                package(project, engine, root)

    def test_inventory_rejects_credentials_and_unsafe_destinations(
        self,
    ) -> None:
        example = json.loads(
            (Path(__file__).parent / "inventory.example.json").read_text()
        )
        mutations = [
            {"private_key": "secret"},
            {"remote_root": "/mydata/../etc"},
            {"port_base": 65530},
            {
                "node1": {
                    "ssh": "-oProxyCommand=bad",
                    "experiment_ip": "192.168.1.2",
                }
            },
            {"node1": {"ssh": "node1", "experiment_ip": "127.0.0.1"}},
            {"node1": {"ssh": "node1", "experiment_ip": "192.168.1.1"}},
        ]
        with tempfile.TemporaryDirectory(prefix="kvc-inventory-") as temporary:
            path = Path(temporary) / "inventory.json"
            path.write_text(json.dumps(example))
            self.assertEqual(load_inventory(path).port_base, 24000)
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    path.write_text(json.dumps(dict(example, **mutation)))
                    with self.assertRaises(ValueError):
                        load_inventory(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
