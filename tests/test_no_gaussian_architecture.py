"""Prevents accidentally resurrecting the retired standalone 3DGS module.

This is a source-integrity guard, NOT a functioning Omega-Mesh-UV test.
"""
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestNoGaussianMainline(unittest.TestCase):
    def test_retired_branch_is_absent(self):
        self.assertFalse((ROOT / "3dgs").exists())
        self.assertFalse((ROOT / "scripts" / "run_3dgs.py").exists())
        upstream = json.loads((ROOT / "configs" / "upstreams.json").read_text())
        manifest = json.loads((ROOT / "configs" / "vendored_manifest.json").read_text())
        self.assertEqual(set(upstream), {"vggt", "2dsugar"})
        for key in ("upstreams", "counts", "files"):
            self.assertNotIn("3dgs", manifest[key])

    def test_run_all_is_no_longer_the_gaussian_pipeline(self):
        """run_all.sh used to drive 3DGS. It must now explain the new stages or refuse."""
        script = ROOT / "scripts" / "run_all.sh"
        text = script.read_text(encoding="utf-8")
        for retired in ("simple_trainer", "run_3dgs", "gsplat"):
            self.assertNotIn(retired, text, f"run_all.sh must not resurrect {retired}")
        # Feed the script on stdin as BYTES: Git Bash rewrites Windows argv paths, and
        # text=True would translate "\n" to "\r\n", which bash rejects at `set -o pipefail`.
        result = subprocess.run(["bash", "-s"], input=text.encode("utf-8"), cwd=str(ROOT),
                                capture_output=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        stderr = result.stderr.decode("utf-8", "replace")
        self.assertIn("retired", stderr)
        self.assertIn("MESH_UV_PIPELINE", stderr)

    def test_architecture_and_paper_docs_exist(self):
        docs = ROOT / "docs"
        for path in ("MESH_UV_PIPELINE.md", "SOTA_2026.md",
                     "VGGT_OMEGA_ADAPTER.md", "DECISION_NO_GAUSSIANS.md",
                     "DIRECT_UV.md", "papers/README.md"):
            self.assertGreater((docs / path).stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
