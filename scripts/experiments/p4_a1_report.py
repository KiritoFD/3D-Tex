#!/usr/bin/env python3
"""Summarise phase-4 A1 carve/mesh artefacts (read-only)."""
from __future__ import annotations

import glob
import json
import statistics as st


def main() -> None:
    for path in sorted(glob.glob("/root/3dtex/outputs/72h/phase4/a1/*/*/geometry/omega/carve_report.json")):
        data = json.load(open(path))
        frames = list(data["frames"].values())
        tag = path.split("a1/")[1].split("/geometry")[0]
        deleted = [f["deleted_fraction_of_confident"] for f in frames]
        print(
            f"{tag:>28} tau={data['tau']:.4f} conf={data['conf_threshold']:.4f} "
            f"surv={data['surviving_fraction']:.3f} del_mean={st.mean(deleted):.3f} "
            f"del_min={min(deleted):.3f} del_max={max(deleted):.3f} "
            f"free={st.mean(f['mean_free_fraction'] for f in frames):.3f} "
            f"sup={st.mean(f['mean_support_fraction'] for f in frames):.3f}"
        )
    print("--- mesh ---")
    for path in sorted(glob.glob("/root/3dtex/outputs/72h/phase4/a1/*/*/mesh/mesh_report.json")):
        print(path.split("a1/")[1].split("/mesh")[0], json.load(open(path)))


if __name__ == "__main__":
    main()
