"""One high-value regression: rethreshold cached distances without rerunning models."""
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


if __name__ == "__main__":
    unittest.main()
