"""Small regressions for foreground, camera export and unknown TSDF boundaries."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from exmesh_adapter import build_colmap, resize_intrinsics
from geometry_masks import load_masks
from tsdf_mesh import extract_surface, fuse
from diagnostics.pose_depth_ablation import depth_normals


class GeometryFixes(unittest.TestCase):
    def test_depth_normal_operator_preserves_a_frontoparallel_plane(self):
        k = np.array([[100.,0,16.],[0,100.,16.],[0,0,1.]])
        normal, valid = depth_normals(np.ones((32,32)), k, radius=2)
        self.assertEqual(int(valid.sum()), 28*28)
        np.testing.assert_allclose(normal[valid, :2], 0, atol=1e-12)
        np.testing.assert_allclose(normal[valid, 2], 1, atol=1e-12)

    def test_unknown_band_does_not_close_surface(self):
        field = np.full((8, 8, 8), np.nan, np.float32)
        field[2:6] = np.arange(4)[:, None, None] - 1.5
        vertices, faces, _ = extract_surface(field, 1.)
        self.assertGreater(len(faces), 0)
        np.testing.assert_allclose(vertices[:, 0], 4.)

    def test_cameras_retain_per_view_intrinsics(self):
        k = np.array([[3402.536558, 0, 3105.5], [0, 3401.461602, 2068.5], [0, 0, 1]])
        scaled = resize_intrinsics(k, (6211, 4137), (1024, 682))
        self.assertAlmostEqual(scaled[0, 0], 560.97, delta=.01)
        np.testing.assert_allclose(scaled[:2, 2], [512, 341])
        with tempfile.TemporaryDirectory() as tmp:
            cameras = [{"_K": np.diag([f, f, 1.]), "_T": np.eye(4), "file": f"{i}.png"}
                       for i, f in enumerate((561., 570.))]
            build_colmap(Path(tmp), cameras, 1024, 682)
            rows = (Path(tmp) / "cameras.txt").read_text().splitlines()[1:]
            self.assertEqual([float(r.split()[4]) for r in rows], [561., 570.])
            self.assertIn("2 PINHOLE", rows[1])

    def test_alpha_maps_to_model_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp)
            (scene / "images").mkdir()
            rgba = np.full((8, 8, 4), 255, np.uint8)
            rgba[:, :4, 3] = 0
            Image.fromarray(rgba).save(scene / "images" / "a.png")
            frame = {"file": "a.png", "model_width": 4, "model_height": 4,
                     "pixel_affine_original_to_model": np.diag([.5, .5, 1.]).tolist()}
            mask = load_masks(scene, [frame], "alpha")[0]
            self.assertFalse(mask[:, :2].any())
            self.assertTrue(mask[:, 2:].all())

    def test_voxel_receives_one_observation_per_frame(self):
        frame = {"depth": np.ones((8, 8)), "K": np.array([[4.,0,4],[0,4.,4],[0,0,1]]),
                 "T": np.eye(4), "rgb": None}
        _, _, weights = fuse(np.array([-.2,-.2,.8]), .1, (4,4,4), .2,
                             [frame], 0, 2, [None], "uniform", 64, 1)
        self.assertEqual(weights.max(), 1.)
        later = dict(frame, depth=np.full((8, 8), 1.1))
        field, _, weights = fuse(np.array([-.2,-.2,.8]), .1, (4,4,4), .2,
                                 [frame, later], 0, 2, [None, None], "uniform", 1, 1)
        self.assertAlmostEqual(float(field[1, 1, 1]), .5, places=5)
        self.assertEqual(weights.max(), 1.)


if __name__ == "__main__":
    unittest.main()
