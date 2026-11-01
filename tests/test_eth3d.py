"""CPU-only tests for ETH3D downloader, COLMAP calibration, and leakage-free split."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from download_eth3d import (
    SCENES, archive_names, build_scene, choose_split, locate_calibration,
    locate_images, read_gt_records,
)


def fake_eth3d_archive(root: Path, count: int = 14, with_geometry: bool = True) -> Path:
    scene = root / "pipes"
    images = scene / "images" / "dslr_images_undistorted"
    calibration = scene / "dslr_calibration_undistorted"
    images.mkdir(parents=True)
    calibration.mkdir()
    (calibration / "cameras.txt").write_text(
        "# Camera list\n1 PINHOLE 100 80 80 80 50 40\n", encoding="utf-8"
    )
    rows = ["# Image list (pose line, observations line)\n"]
    for i in range(count):
        name = f"DSC_{i:04d}.JPG"
        (images / name).write_bytes(b"fake-jpeg-" + bytes([i]))
        rows.extend(
            [f"{i+1} 1 0 0 0 {i}.0 0 0 1 dslr_images/{name}\n", "\n"]
        )
    (calibration / "images.txt").write_text("".join(rows), encoding="utf-8")
    if with_geometry:
        scan = scene / "dslr_scan_eval"
        scan.mkdir()
        (scan / "scan_eval.ply").write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
    return root


class TestEth3D(unittest.TestCase):
    def test_archives(self):
        self.assertEqual(len(SCENES), 3)
        self.assertEqual(archive_names("pipes", True), [
            "pipes_dslr_undistorted.7z", "pipes_dslr_scan_eval.7z"
        ])
        self.assertEqual(len(archive_names("relief", False)), 1)

    def test_calibration_and_split(self):
        with tempfile.TemporaryDirectory() as d:
            stage = fake_eth3d_archive(Path(d))
            cameras, images = locate_calibration(stage)
            self.assertEqual(cameras.name, "cameras.txt")
            found = locate_images(stage)
            self.assertEqual(len(found), 14)
            records = read_gt_records(images)
            self.assertEqual(len(records), 14)
            train, test = choose_split(records, 5)
            self.assertEqual((len(train), len(test)), (12, 2))
            self.assertFalse(set(train) & set(test))
            self.assertTrue(all(x.startswith("DSC_") for x in test))

    def test_prepare_isolated_and_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            stage = fake_eth3d_archive(root / "extracted")
            dest = root / "eth3d_pipes"
            manifest = build_scene(stage, "pipes", True, 5, dest, {"test": "hash"})
            train = list((dest / "images").glob("*"))
            heldout = list((dest / "heldout" / "images").glob("*"))
            self.assertEqual((len(train), len(heldout)), (12, 2))
            self.assertEqual(len(list((dest / "ground_truth" / "scan_eval").glob("*.ply"))), 1)
            self.assertFalse({x.name for x in train} & {x.name for x in heldout})
            self.assertFalse((dest / "sparse").exists())
            self.assertTrue((dest / "ground_truth" / "calibration" / "poses.json").is_file())
            again = build_scene(stage, "pipes", True, 5, dest, {"test": "hash"})
            self.assertEqual(manifest, again)

    def test_missing_cameras_must_fail(self):
        with tempfile.TemporaryDirectory() as d:
            stage = fake_eth3d_archive(Path(d), with_geometry=False)
            (stage / "pipes" / "dslr_calibration_undistorted" / "images.txt").unlink()
            with self.assertRaises(FileNotFoundError):
                locate_calibration(stage)

    def test_archive_errors(self):
        with self.assertRaises(ValueError):
            archive_names("large_dataset", True)
        with self.assertRaises(ValueError):
            choose_split([], 2)


if __name__ == "__main__":
    unittest.main()
