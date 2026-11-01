"""CPU-only sanity: original VGGT's COLMAP adapter and optional legacy wrapper.

The experiment is deliberately not the proposed VGGT-Omega -> Mesh+UV system.
No training, GPU model inference, rendering, or texture is executed.
"""
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_vggt
import run_2dsugar
from prepare_scene import prepare
from scene_scale import robust_extent


def mock_colmap(scene: Path) -> None:
    sparse = scene / "sparse"
    sparse.mkdir(exist_ok=True)
    (sparse / "cameras.bin").write_bytes(b"synthetic-camera-fixture")
    (sparse / "images.bin").write_bytes(b"synthetic-image-fixture")
    with (sparse / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 32))
        for i in range(32):
            handle.write(struct.pack("<Q3d3BdQ", i + 1, float(i),
                                     i * 0.2, -i * 0.1, 100, 130, 150, 0.0, 0))


class TestRetainedCPULegacyWrappers(unittest.TestCase):
    def test_original_vggt_colmap_export_bridge_and_optional_legacy_sugar(self):
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
            for path in ("cameras.bin", "images.bin", "points3D.bin"):
                self.assertTrue((bridge / path).is_file())
            self.assertGreater(robust_extent(bridge), 10)
            self.assertEqual(prepare(scene), bridge)

            def capture(stage, name, argv, cwd, dry_run=False):
                commands.append((stage, list(argv)))

            # Historical 2D-SuGaR remains opt-in, NOT the current research mainline.
            with patch.object(run_2dsugar, "scene_path", return_value=scene), \
                 patch.object(run_2dsugar, "execute", side_effect=capture), \
                 patch.object(sys, "argv", ["run_2dsugar.py", "--scene", "fixture",
                                           "--iterations", "20", "--refine-iterations", "10"]):
                run_2dsugar.main()
            self.assertEqual([x[0] for x in commands], ["vggt", "2dsugar"])
            s = commands[-1][1]
            self.assertEqual(s[s.index("-s") + 1], str(scene))
            self.assertEqual(s[s.index("--export_uv_textured_mesh") + 1], "True")
            self.assertEqual(s[s.index("--initialization_prior") + 1], "")
            self.assertNotIn("ground_truth", " ".join(s))


if __name__ == "__main__":
    unittest.main()
