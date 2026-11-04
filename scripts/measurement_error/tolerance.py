"""Aggregate both error branches into one tolerance table.

Reads the frozen per-case tables produced by evaluate.py, the optimization
logs and the zero-noise repeatability record. It does not create new metrics:
it only groups, pairs and compares existing columns.

Definitions stay separated because they are different physical quantities:
  image-plane  : object_warp  (px -> mm at local GT depth)
  camera       : camera_jitter (actual camera-centre translation)
  depth        : iid / view_bias (camera z error, independent or image-shared)
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PER_CASE = [
    "chamfer_observed_mm", "chamfer_full_mm", "normal_deg",
    "normal_flat_deg", "normal_edge_deg",
    "fscore@1mm", "precision@1mm", "recall@1mm",
    "hole_fill", "thin_proxy_recall@1mm", "realized_rms_mm",
]
OPT_COLUMNS = ["final_loss", "delta_mean_mm", "delta_p90_mm", "delta_max_mm",
               "seconds", "peak_vram_gb"]
PRIMARY = ["chamfer_observed_mm", "normal_deg", "fscore@1mm", "hole_fill"]

DEFINITIONS = [
    ("image_plane", "object_warp", "RGB", "object_warp",
     "image-plane localization displacement, px converted at local GT depth"),
    ("camera_translation", "camera_jitter", "RGB", "camera_jitter",
     "actual camera-centre translation, orientation held fixed"),
    ("depth_error_iid", "iid", "external", "iid",
     "per-pixel independent camera-z depth error"),
    ("depth_error_shared", "view_bias", "external", "view_bias",
     "one shared camera-z bias per image (not averaged by multiple views)"),
]


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def number(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def optimization_diagnostics(out_root, case, mm_per_source_unit):
    """Final loss and vertex displacement from the frozen M2 log.

    ``delta_*_voxels`` is in lattice units; the lattice edge is
    ``voxel_size`` in source units, so mm = voxels * voxel_size * mm_per_unit.
    """
    log = Path(out_root) / "m2" / case / "m2_log.json"
    if not log.exists():
        return {}
    data = json.loads(log.read_text())
    voxel_mm = data["args"]["voxel_size"] * mm_per_source_unit
    last = data["iters"][-1]
    return {"final_loss": last["loss"],
            "delta_mean_mm": last["delta_mean_voxels"] * voxel_mm,
            "delta_p90_mm": last["delta_p90_voxels"] * voxel_mm,
            "delta_max_mm": last["delta_max_voxels"] * voxel_mm,
            "seconds": data["seconds"], "peak_vram_gb": data["peak_vram_gb"]}


def pooled(rows, column):
    values = [number(row.get(column)) for row in rows]
    return [v for v in values if v is not None]


def describe(values):
    if not values:
        return {}
    return {"mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values), "n": len(values)}


def build(m2_eval, rgb_eval, m2_out, repeatability, mm_per_source_unit):
    records, references = [], {}
    raw_groups = {}
    for name in ("initial", "no_external"):
        rows = [row for row in m2_eval if row["case"] == name]
        if rows:
            item = {"case": name, "status": rows[0]["status"]}
            for column in PER_CASE:
                item[column] = number(rows[0].get(column))
            item.update(optimization_diagnostics(m2_out, name, mm_per_source_unit)
                        if name == "no_external" else {})
            references[name] = item
    clean_rows = [row for row in m2_eval if number(row.get("target_rms_mm")) == 0.0]
    for definition, mode, family, source_mode, percent in DEFINITIONS:
        rows = m2_eval if family == "external" else rgb_eval
        rows = [row for row in rows if row.get("noise_mode") == source_mode]
        levels = sorted({number(row["target_rms_mm"]) for row in rows
                         if number(row.get("target_rms_mm")) is not None})
        for level in levels:
            group = [row for row in rows
                     if number(row.get("target_rms_mm")) == level and row["status"] == "complete"]
            if not group:
                continue
            record = {"definition": definition, "mode": mode, "family": family,
                      "definition_label": percent, "level_mm": level,
                      "n_complete": len(group),
                      "n_expected": 3, "seeds": ",".join(sorted(row["seed"] for row in group))}
            for column in PER_CASE:
                stats = describe(pooled(group, column))
                if stats:
                    record[column + "_mean"] = stats["mean"]
                    record[column + "_std"] = stats["std"]
                    record[column + "_min"] = stats["min"]
                    record[column + "_max"] = stats["max"]
            if family == "external":
                diags = [optimization_diagnostics(m2_out, row["case"], mm_per_source_unit)
                         for row in group]
                for column in OPT_COLUMNS:
                    stats = describe([d[column] for d in diags if column in d])
                    if stats:
                        record[column + "_mean"] = stats["mean"]
                        record[column + "_std"] = stats["std"]
            records.append(record)
            raw_groups[(definition, level)] = group

    floors = {"metric": {column: describe(pooled(clean_rows, column)) for column in PER_CASE},
              "vertex_displacement": repeatability}
    by_definition = {}
    for record in records:
        by_definition.setdefault(record["definition"], []).append(record)

    # Each definition gets its own zero-noise control: pooling iid and view_bias is
    # valid (their rms=0 cases are the same runs), but the RGB branch has a different
    # pipeline and must not inherit the optimizer's band.
    for definition, rows in by_definition.items():
        if definition.startswith("depth_error"):
            control = clean_rows
        else:
            control = raw_groups.get((definition, 0.0), [])
        for column in PER_CASE:
            samples = pooled(control, column)
            if not samples:
                continue
            spread = describe(samples)
            pooled_stats = floors["metric"].get(column, {})
            spread["band"] = max(2 * spread["std"], spread["max"] - spread["min"],
                                 2 * pooled_stats.get("std", 0.0),
                                 pooled_stats.get("max", 0.0) - pooled_stats.get("min", 0.0))
            floors.setdefault(definition, {})[column] = spread

    for definition, rows in by_definition.items():
        clean = next((r for r in rows if r["level_mm"] == 0.0), None)
        for record in rows:
            if not clean:
                continue
            for column in PRIMARY:
                key = column + "_mean"
                if key not in record or key not in clean:
                    continue
                floor = floors.get(definition, {}).get(column, {})
                band = floor.get("band", 0.0)
                record["delta_" + column] = record[key] - clean[key]
                record["floor_" + column] = band
                record["within_floor_" + column] = abs(record["delta_" + column]) <= max(band, 1e-12)
            if "chamfer_observed_mm_mean" in record and "chamfer_observed_mm_mean" in clean:
                ratio = record["chamfer_observed_mm_mean"] / clean["chamfer_observed_mm_mean"]
                record["chamfer_observed_ratio"] = ratio
                record["chamfer_observed_max_ratio"] = record["chamfer_observed_mm_max"] / clean["chamfer_observed_mm_mean"]
            if "fscore@1mm_mean" in record and "fscore@1mm_mean" in clean:
                record["fscore@1mm_ratio"] = record["fscore@1mm_mean"] / clean["fscore@1mm_mean"]
            # Two verdicts, both floor-relative and both one-sided on degradation.
            # A change in the improving direction is a change, not a cost.
            worsening = {}
            for column in PRIMARY:
                delta = record.get("delta_" + column)
                band = record.get("floor_" + column)
                if delta is None or band is None:
                    continue
                worsening[column] = -delta if column == "fscore@1mm" else delta
            record["verdict_no_degradation"] = all(
                worse <= 2 * record["floor_" + column]
                for column, worse in worsening.items()) if worsening else False
            record["verdict_acceptable"] = bool(
                record.get("chamfer_observed_max_ratio", 9) <= 1.10
                and record.get("fscore@1mm_ratio", 0) >= 0.95
                and record.get("delta_normal_deg", -9) <= 3 * record.get("floor_normal_deg", 0.0))
            record["marginal_floor"] = any(
                abs(record["delta_" + column]) > record["floor_" + column] for column in worsening)
    return records, references, floors


def write_table(records, path):
    fields = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def print_summary(records, floors):
    print("\nNoise floor: zero-noise repeats (6 M2 runs, identical inputs)")
    for column in ("chamfer_observed_mm", "chamfer_full_mm", "normal_deg",
                   "fscore@1mm", "hole_fill", "thin_proxy_recall@1mm"):
        stats = floors["metric"].get(column, {})
        print(f"  {column:24s} mean {stats.get('mean', float('nan')):.4f} "
              f"std {stats.get('std', float('nan')):.4f} "
              f"range [{stats.get('min', float('nan')):.4f}, {stats.get('max', float('nan')):.4f}] "
              f"band {max(2*stats.get('std', 0.0), stats.get('max', 0.0)-stats.get('min', 0.0)):.4f}")
    print("\nPer-definition zero-noise control band (max of 2 sigma and full range)")
    for definition, _, _, _, _ in DEFINITIONS:
        table = floors.get(definition)
        if not table or "chamfer_observed_mm" not in table:
            continue
        sample = table["chamfer_observed_mm"]
        print(f"  {definition:26s} n={sample.get('n')} mean {sample.get('mean'):.4f} "
              f"band {table['chamfer_observed_mm']['band']:.4f} | normal band "
              f"{table.get('normal_deg', {}).get('band', float('nan')):.3f} deg")
    print("\nTolerance table")
    header = f"{'definition':26s} {'mm':>5s} {'n':>2s} {'obsCh':>8s} {'±':>7s} {'ratio':>6s} " \
             f"{'dCh':>7s} {'norm':>7s} {'dN':>6s} {'F1':>6s} {'P/R':>11s} {'hole':>6s} {'thin':>6s} {'no-deg':>6s} {'ok':>4s}"
    print(header)
    for record in records:
        def fmt(key, digits=3):
            value = record.get(key)
            return "-" if value is None else f"{value:.{digits}f}"
        pr = "-"
        if record.get("precision@1mm_mean") is not None:
            pr = f"{record['precision@1mm_mean']:.3f}/{record['recall@1mm_mean']:.3f}"
        print(f"{record['definition']:26s} {record['level_mm']:5.2f} {record['n_complete']:2d} "
              f"{fmt('chamfer_observed_mm_mean')} {fmt('chamfer_observed_mm_std',4)} "
              f"{fmt('chamfer_observed_ratio')} {fmt('delta_chamfer_observed_mm',4):>7s} "
              f"{fmt('normal_deg_mean')} {fmt('delta_normal_deg',3):>6s} "
              f"{fmt('fscore@1mm_mean')} {pr:>11s} {fmt('hole_fill_mean')} "
              f"{fmt('thin_proxy_recall@1mm_mean')} "
              f"{str(record.get('verdict_no_degradation')):>6s} {str(record.get('verdict_acceptable')):>4s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/measurement_error.json")
    parser.add_argument("--out", default="outputs/measurement_error_fullrun/tolerance_table.csv")
    parser.add_argument("--repeatability",
                        default="outputs/measurement_error_fullrun/repeatability.json")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    m2_eval = read_csv(ROOT / config["out"] / "m2_eval" / "results.csv")
    rgb_eval = read_csv(ROOT / config["rgb_sweep"]["out"] / "rgb_eval" / "results.csv")
    repeatability = json.loads((ROOT / args.repeatability).read_text())
    master = ROOT / config["master"]
    mm_per_source_unit = json.loads((master / "scale.json").read_text())["mm_per_source_unit"]
    records, _, floors = build(m2_eval, rgb_eval, ROOT / config["out"], repeatability,
                               mm_per_source_unit)
    write_table(records, ROOT / args.out)
    print_summary(records, floors)
    print("\nwritten", ROOT / args.out)


if __name__ == "__main__":
    main()
