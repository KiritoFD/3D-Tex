"""CPU-only guards for the two coordinate bugs that made every mesh look broken.

Both were "self-consistent but wrong": the intrinsics matched the depth, and the filter used
the same wrong K, so nothing looked off until compared with ETH3D ground truth. These tests
pin the arithmetic so a regression fails loudly instead of degrading quietly.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# ETH3D kicker frame 0, measured from the official undistorted calibration.
GT_FX, GT_CX, GT_CY = 3410.34, 3121.33, 2067.07
ORIGINAL_WH = (6211, 4137)
MODEL_WH = (624, 416)
K_MODEL = np.array([[341.76, 0.0, 312.0], [0.0, 342.03, 208.0], [0.0, 0.0, 1.0]])


def original_from_model(k_model: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Same formula as scripts/adapters/vggt_omega_geometry.py."""
    return np.array([
        [k_model[0, 0] / affine[0, 0], 0.0, (k_model[0, 2] - affine[0, 2]) / affine[0, 0]],
        [0.0, k_model[1, 1] / affine[1, 1], (k_model[1, 2] - affine[1, 2]) / affine[1, 1]],
        [0.0, 0.0, 1.0]])


class TestIntrinsicConversion(unittest.TestCase):
    def setUp(self):
        sx = MODEL_WH[0] / ORIGINAL_WH[0]
        sy = MODEL_WH[1] / ORIGINAL_WH[1]
        self.affine = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])

    def test_conjugation_is_a_noop_and_must_not_be_used(self):
        """A^-1 K A leaves focal unchanged: that is exactly the bug we are guarding against."""
        conjugated = np.linalg.inv(self.affine) @ K_MODEL @ self.affine
        self.assertAlmostEqual(conjugated[0, 0], K_MODEL[0, 0], places=9)
        self.assertLess(conjugated[0, 0] / GT_FX, 0.2, "conjugation gives a 10x-too-small focal")

    def test_scaled_conversion_matches_ground_truth(self):
        k_original = original_from_model(K_MODEL, self.affine)
        self.assertAlmostEqual(k_original[0, 0] / GT_FX, 1.0, delta=0.01)
        self.assertAlmostEqual(k_original[0, 2] / GT_CX, 1.0, delta=0.01)

    def test_affine_maps_image_width_onto_model_width(self):
        self.assertAlmostEqual(self.affine[0, 0] * ORIGINAL_WH[0], MODEL_WH[0], places=6)

    def test_central_crop_offset_is_carried_into_the_principal_point(self):
        cropped_width = ORIGINAL_WH[0] - 200
        sx = MODEL_WH[0] / cropped_width
        affine = np.array([[sx, 0.0, -100 * sx], [0.0, MODEL_WH[1] / ORIGINAL_WH[1], 0.0],
                           [0.0, 0.0, 1.0]])
        k_original = original_from_model(K_MODEL, affine)
        # the crop shifts the model centre, so the recovered cx must move with it
        self.assertLess(k_original[0, 2], GT_CX)


class TestIntrinsicsForDepthGrid(unittest.TestCase):
    def test_depth_grid_uses_k_model_not_k_original(self):
        """Depth maps live on the model grid; using K_original collapsed consistency to 0."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import inspect
        import consistency_filter
        source = inspect.getsource(consistency_filter.load_geometry)
        self.assertIn('frame["K_model"]', source)
        self.assertNotIn('frame["K_original"]', source)


if __name__ == "__main__":
    unittest.main()
