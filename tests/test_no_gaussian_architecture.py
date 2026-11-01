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

    def test_legacy_run_all_fails_closed(self):
        result = subprocess.run(["bash", str(ROOT / "scripts" / "run_all.sh")],
                                capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("retired", result.stderr)
        self.assertIn("NOT", result.stderr.upper().replace("NOT YET", "NOT"))

    def test_architecture_and_paper_docs_exist(self):
        docs = ROOT / "docs"
        for path in ("MESH_UV_PIPELINE.md", "SOTA_2026.md",
                     "VGGT_OMEGA_ADAPTER.md", "DECISION_NO_GAUSSIANS.md",
                     "DIRECT_UV.md", "papers/README.md"):
            self.assertGreater((docs / path).stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
