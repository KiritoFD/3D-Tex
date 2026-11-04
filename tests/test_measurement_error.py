"""Regressions for the measurement-error framework.

v1: rethreshold cached distances without rerunning models.
v2/R0: byte-identical zero-noise scans, mm/px/camera unit contracts, paired
sampling, locked registration for non-ideal calibration, and the prediction
mode GT guard.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from measurement_error.evaluate import table


class MeasurementTable(unittest.TestCase):
    def test_threshold_edit_preserves_raw_data_and_pending_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = []
            for level in (0, 1):
                case = root/f"iid_rms{level}_s42"
                case.mkdir()
                (case/"manifest.json").write_text(json.dumps({"noise_mode": "iid", "seed": 42,
                    "target_rms_mm": level, "realized_rms_mm": level}))
                catalog.append({"name": case.name, "case": case.as_posix()})
            (root/"catalog.json").write_text(json.dumps(catalog))
            result = root/"out/m2_eval/iid_rms0_s42"
            result.mkdir(parents=True)
            np.savez_compressed(result/"distances.npz", accuracy_mm=[0., .6, 2.],
                completeness_mm=[0., .5, 1.5], observed_mm=[0., .25, .75],
                normal_deg=[10., 20., 30.], thin_depth_error_mm=[.4, 1.2, np.inf])
            (result/"metadata.json").write_text(json.dumps({"hole_fill_rate": .1}))
            raw_hash = hashlib.sha256((result/"distances.npz").read_bytes()).hexdigest()
            config = {"root": str(root), "out": str(root/"out"), "noise_modes": ["iid"],
                "noise_rms_mm": [0, 1], "seeds": [42, 43, 44], "fscore_thresholds_mm": [.5, 1., 2.]}
            table(config, "m2")
            config["fscore_thresholds_mm"] = [.75]
            table(config, "m2")
            with (root/"out/m2_eval/results.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = {r["case"]: r for r in csv.DictReader(stream)}
            self.assertAlmostEqual(float(rows["iid_rms0_s42"]["fscore@0.75mm"]), 2/3)
            self.assertNotIn("fscore@0.5mm", rows["iid_rms0_s42"])
            self.assertEqual(rows["iid_rms1_s42"]["status"], "pending")
            self.assertEqual(rows["iid_rms1_s42"]["chamfer_full_mm"], "")
            self.assertEqual(hashlib.sha256((result/"distances.npz").read_bytes()).hexdigest(), raw_hash)
            with (root/"out/m2_eval/grouped.csv").open(encoding="utf-8-sig", newline="") as stream:
                groups = list(csv.DictReader(stream))
            self.assertEqual((groups[0]["n_complete"], groups[0]["n_expected"]), ("1", "3"))


def orbit_pose(index, count=4, radius=1.15):
    """Rigid look-at camera on a circle; returns T_world_to_cam."""
    azimuth = 2*np.pi*index/count
    eye = np.array([radius*np.sin(azimuth), .22, radius*np.cos(azimuth)])
    forward = -eye/np.linalg.norm(eye)
    right = np.cross(forward, np.array([0., 1., 0.]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, forward, eye
    return np.linalg.inv(c2w)


class ZeroNoiseScan(unittest.TestCase):
    def test_zero_rms_scan_reproduces_gt_depth_bytes(self):
        from measurement_error import scan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = root/"master/geometry/gt_ray256"
            for folder in ("depth", "foreground"):
                (truth/folder).mkdir(parents=True)
            frames = []
            for index in range(3):
                depth = np.full((256, 256), 470.+index, np.float32)
                np.save(truth/"depth"/f"r_{index}.npy", depth)
                np.save(truth/"foreground"/f"r_{index}.npy", depth > 0)
                frames.append({"file": f"r_{index}.png", "frame": f"r_{index}.png",
                               "model_width": 256, "model_height": 256,
                               "K_model": [[400., 0, 127.5], [0, 400., 127.5], [0, 0, 1]],
                               "T_world_to_cam": orbit_pose(index, 3).tolist()})
            (truth/"cameras.json").write_text(json.dumps({"frames": frames}))
            (truth/"ray_audit.json").write_text(json.dumps({"max_error_mm": .001}))
            (root/"master/scale.json").write_text(json.dumps(
                {"mm_per_source_unit": 2., "source_sha256": "frozen"}))
            rgb = root/"rgb"
            (rgb/"images").mkdir(parents=True)
            for frame in frames:
                (rgb/"images"/frame["file"]).write_bytes(b"rendered")
            config = {"master": str(root/"master"), "root": str(root/"cases"), "rgb_run": str(rgb),
                      "observation_resolution": 256, "noise_modes": ["iid"],
                      "noise_rms_mm": [0., 1.], "seeds": [42]}
            scan.prepare(config)
            zero = root/"cases/scans/iid_rms0_s42"
            for frame in frames:
                stem = Path(frame["file"]).stem
                produced = hashlib.sha256((zero/"depth"/f"{stem}.npy").read_bytes()).hexdigest()
                reference = hashlib.sha256((truth/"depth"/f"{stem}.npy").read_bytes()).hexdigest()
                self.assertEqual(produced, reference, f"zero-noise scan changed {stem}")
            manifest = json.loads((zero/"manifest.json").read_text())
            self.assertEqual(manifest["realized_rms_mm"], 0.)
            noisy = json.loads((root/"cases/scans/iid_rms1_s42"/"manifest.json").read_text())
            self.assertAlmostEqual(noisy["realized_rms_mm"], 1., delta=.02)
            self.assertAlmostEqual(noisy["mean_error_mm"], 0., delta=.02)
            changed = (root/"cases/scans/iid_rms1_s42/depth/r_0.npy").read_bytes()
            self.assertNotEqual(hashlib.sha256(changed).hexdigest(),
                                hashlib.sha256((truth/"depth/r_0.npy").read_bytes()).hexdigest())


class CalibrationContract(unittest.TestCase):
    def setUp(self):
        self.poses = np.stack([orbit_pose(index) for index in range(4)])

    def test_paired_samples_scale_exactly(self):
        from measurement_error import calibration
        small, record_small = calibration.perturb_poses(self.poses, 1., 0., "independent", 7, 1.)
        large, record_large = calibration.perturb_poses(self.poses, 2., 0., "independent", 7, 1.)
        base = calibration.camera_centers(self.poses)
        np.testing.assert_allclose((calibration.camera_centers(large)-base),
                                   2*(calibration.camera_centers(small)-base), atol=1e-12)
        self.assertAlmostEqual(record_small["translation_rms_mm"], 1., places=9)
        self.assertAlmostEqual(record_large["translation_rms_mm"], 2., places=9)

    def test_distributions(self):
        from measurement_error import calibration
        base = calibration.camera_centers(self.poses)
        bias, record = calibration.perturb_poses(self.poses, 1., 0., "bias", 7, 1.)
        delta = calibration.camera_centers(bias)-base
        np.testing.assert_allclose(delta, np.repeat(delta[:1], len(delta), 0), atol=1e-12)
        self.assertAlmostEqual(record["translation_rms_mm"], 1., places=9)
        drift, record = calibration.perturb_poses(self.poses, 1., 0., "drift", 7, 1.)
        delta = calibration.camera_centers(drift)-base
        self.assertAlmostEqual(record["translation_rms_mm"], 1., places=9)
        self.assertAlmostEqual(float(np.abs(delta).mean() <= np.abs(delta).max()), 1.)
        self.assertTrue(np.all(np.diff(np.linalg.norm(delta, axis=1)) >= 0),
                        "drift must grow with the view order")

    def test_rotation_keeps_the_optical_centre(self):
        from measurement_error import calibration
        rotated, record = calibration.perturb_poses(self.poses, 0., .5, "independent", 3, 1.)
        np.testing.assert_allclose(calibration.camera_centers(rotated),
                                   calibration.camera_centers(self.poses), atol=1e-12)
        self.assertAlmostEqual(record["rotation_rms_deg"], .5, places=9)

    def test_mm_px_and_camera_units_agree(self):
        from measurement_error import calibration
        k = np.array([[400., 0, 255.5], [0, 400., 255.5], [0, 0, 1]])
        self.assertAlmostEqual(calibration.equivalent_shift_px(k, .5, 470.), .4255, places=3)
        self.assertAlmostEqual(calibration.equivalent_shift_px(k, 470./400., 470.), 1., places=9)
        # 0.1 deg at 470 mm is 0.8206 mm laterally, i.e. 470*tan(0.1deg)
        self.assertAlmostEqual(470*np.tan(np.deg2rad(.1)), .8206, places=3)
        self.assertLess(470*np.tan(np.deg2rad(.1)), 1.*470./400.)  # still sub-pixel at f=400
        focal = calibration.perturb_intrinsics(k, .005, 0.)
        self.assertAlmostEqual(focal[0, 0]/k[0, 0], 1.005, places=12)
        # a 0.5% focal error moves a pixel at radius 200 px by exactly 1 px
        self.assertAlmostEqual(.005*200., 1., places=9)
        shifted = calibration.perturb_intrinsics(k, 0., 2.)
        np.testing.assert_allclose(shifted[:2, 2], k[:2, 2]+2.)


class LockedRegistration(unittest.TestCase):
    def build(self, root, calibration_mode="non_ideal", translation=0., rotation=0., distribution="bias", seed=5):
        from measurement_error import calibration
        predictions, gt = [], []
        for index in range(4):
            pose = orbit_pose(index)
            gt.append(pose)
            camera = np.linalg.inv(pose)
            camera[:3, 3] = 1.1*camera[:3, 3]+np.array([.03, -.07, .05])
            predictions.append(np.linalg.inv(camera))
        predictions, gt = np.stack(predictions), np.stack(gt)
        frames = [{"frame": f"r_{index}.png", "model_width": 512, "model_height": 512,
                   "K_model": [[443.1, 0, 255.5], [0, 443.1, 255.5], [0, 0, 1]],
                   "T_world_to_cam": predictions[index].tolist()} for index in range(len(predictions))]
        omega = root/"rgb/geometry/omega"
        omega.mkdir(parents=True, exist_ok=True)
        (omega/"cameras.json").write_text(json.dumps({"frames": frames}))
        master = root/"master"
        (master/"geometry/gt_ray256").mkdir(parents=True, exist_ok=True)
        (master/"geometry/gt_ray256/cameras.json").write_text(json.dumps({"frames": [
            {"frame": frame["frame"], "model_width": 512, "model_height": 512,
             "K_model": frame["K_model"], "T_world_to_cam": gt[index].tolist()}
            for index, frame in enumerate(frames)]}))
        (master/"scale.json").write_text(json.dumps({"mm_per_source_unit": .214}))
        config = {"rgb_run": str(root/"rgb"), "master": str(master),
                  "observation_resolution": 256, "model_resolution": 512}
        return calibration.build(config, translation, rotation, distribution, seed,
                                 calibration_mode=calibration_mode)

    def test_injected_error_survives_the_fixed_registration(self):
        from measurement_error import calibration
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ideal = self.build(root, "ideal")
            injected = self.build(root, translation=2., distribution="bias")
            gt = [np.asarray(frame["T_world_to_cam"]) for frame in
                  json.loads((root/"master/geometry/gt_ray256/cameras.json").read_text())["frames"]]
            used = np.stack([np.asarray(frame["T_world_to_cam"]) for frame in injected["frames"].values()])
            ideal_used = np.stack([np.asarray(frame["ideal_pose"]) for frame in injected["frames"].values()])
            np.testing.assert_allclose(calibration.align_poses(list(ideal_used), gt)[0],
                                       ideal_used, atol=1e-9)
            self.assertAlmostEqual(injected["injected"]["translation_rms_mm"], 2., places=6)
            scale = injected["scale_mm_per_source_unit"]
            # After the *fixed* ideal registration the algorithm camera is still 2 mm
            # away from the GT camera; a refit would have reduced this to zero.
            residual = np.linalg.norm(calibration.camera_centers(used)-calibration.camera_centers(gt),
                                      axis=1)*scale
            np.testing.assert_allclose(residual, np.full(len(residual), 2.), atol=1e-6)
            self.assertAlmostEqual(float(np.linalg.norm(
                calibration.camera_centers(used)-calibration.camera_centers(ideal_used), axis=1).mean()), 2./scale,
                places=6)

    def test_modes_are_never_mixed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.build(Path(directory), "ideal", translation=1.)

    def test_prediction_mode_rejects_gt_fallback_and_calibration(self):
        from types import SimpleNamespace
        from mesh_opt.inputs import load_views
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            omega = root/"omega"
            omega.mkdir()
            (omega/"cameras.json").write_text(json.dumps({"frames": [
                {"frame": "r_0.png", "model_width": 512, "model_height": 512,
                 "K_model": [[443.1, 0, 255.5], [0, 443.1, 255.5], [0, 0, 1]],
                 "pixel_affine_original_to_model": np.eye(3).tolist(),
                 "T_world_to_cam": np.eye(4).tolist()}]}))
            args = SimpleNamespace(omega=str(omega), camera_mode="prediction", mask_dir=None,
                                   images=None, resolution=256, lambda_colmap=0, lambda_external=0)
            with self.assertRaises(ValueError):
                load_views(args, "cpu")
            args.calibration = str(root/"missing.json")
            with self.assertRaises(ValueError):
                load_views(args, "cpu")


if __name__ == "__main__":
    unittest.main()
