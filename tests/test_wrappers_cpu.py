"""Synthetic CPU-only integration of the VGGT -> COLMAP -> two-wrapper flow.

This test stubs VGGT output and GPU optimization. It does not run any ML model.
"""
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_vggt
import run_3dgs
import run_2dsugar
from prepare_scene import prepare
from scene_scale import robust_extent


def mock_colmap(scene):
    sparse = scene / "sparse"
    sparse.mkdir(exist_ok=True)
    (sparse / "cameras.bin").write_bytes(b"synthetic-camera-fixture")
    (sparse / "images.bin").write_bytes(b"synthetic-image-fixture")
    with (sparse / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 32))
        for i in range(32):
            handle.write(struct.pack("<Q3d3BdQ", i + 1, float(i),
                                     i * 0.2, i * -0.1, 100, 130, 150, 0.0, 0))


class TestCPUPipelineWrappers(unittest.TestCase):
    def test_synthetic_branch_wiring(self):
        with tempfile.TemporaryDirectory() as d:
            scene = Path(d) / "scene"
            images = scene / "images"
            images.mkdir(parents=True)
            for i in range(4):
                (images / f"{i:02d}.png").write_bytes(b"fixture")
            commands = []

            def fake_vggt(stage, name, argv, cwd, dry_run=False):
                commands.append((stage, list(argv)))
                mock_colmap(scene)

            with patch.object(run_vggt, "scene_path", return_value=scene), \
                 patch.object(run_vggt, "execute", side_effect=fake_vggt), \
                 patch.object(sys, "argv", ["run_vggt.py", "--scene", "fixture", "--ba"]):
                run_vggt.main()
            self.assertEqual(commands[0][0], "vggt")
            self.assertIn("--use_ba", commands[0][1])
            bridge = scene / "sparse" / "0"
            self.assertTrue(all((bridge / f).is_file() for f in
                                ("cameras.bin", "images.bin", "points3D.bin")))
            self.assertGreater(robust_extent(bridge), 10)
            self.assertEqual(prepare(scene), bridge)

            def capture(stage, name, argv, cwd, dry_run=False):
                commands.append((stage, list(argv)))

            with patch.object(run_3dgs, "scene_path", return_value=scene), \
                 patch.object(run_3dgs, "execute", side_effect=capture), \
                 patch.object(sys, "argv", ["run_3dgs.py", "--scene", "fixture",
                                           "--steps", "20", "--data-factor", "4"]):
                run_3dgs.main()
            with patch.object(run_2dsugar, "scene_path", return_value=scene), \
                 patch.object(run_2dsugar, "execute", side_effect=capture), \
                 patch.object(sys, "argv", ["run_2dsugar.py", "--scene", "fixture",
                                           "--iterations", "20", "--refine-iterations", "10"]):
                run_2dsugar.main()

            self.assertEqual([x[0] for x in commands], ["vggt", "3dgs", "2dsugar"])
            g = commands[1][1]
            s = commands[2][1]
            self.assertIn("--init_type", g)
            self.assertEqual(g[g.index("--init_type") + 1], "sfm")
            self.assertEqual(g[g.index("--data_dir") + 1], str(scene))
            self.assertEqual(s[s.index("-s") + 1], str(scene))
            self.assertEqual(s[s.index("--export_uv_textured_mesh") + 1], "True")
            self.assertEqual(s[s.index("--initialization_prior") + 1], "")
            self.assertNotIn("ground_truth", " ".join(g + s))


if __name__ == "__main__":
    unittest.main()
