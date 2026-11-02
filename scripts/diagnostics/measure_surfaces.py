"""Blender: sample-to-triangle diagnostic distances against the source scene."""
import argparse
import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ply_to_obj import read_ply
from diagnostics.lego_gt_depth import source_bvh


def distances(tree, points):
    values = [tree.find_nearest(tuple(p)) for p in points]
    return (np.asarray([v[3] for v in values], dtype=np.float64),
            np.asarray([v[2] for v in values], dtype=np.int64))


def summarize(values, voxel):
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "mean_in_voxels": float(values.mean()/voxel)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/.cache/nerf-syn/lego_source.blend")
    parser.add_argument("--out", default="deliverables/lego_p0")
    parser.add_argument("--case", default=None, help="measure one additional case and retain the other results")
    parser.add_argument("--mesh", default=None, help="optional external OBJ/ASCII PLY baseline")
    parser.add_argument("--mesh-is-omega", action="store_true")
    parser.add_argument("--source-flags", default=None,
                        help="optional per-source-triangle max-dihedral array (.npy) for structure-split metrics")
    parser.add_argument("--source-flags-threshold", type=float, default=40.0,
                        help="dihedral degrees above which a source triangle counts as sharp structure")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if args.mesh and not args.case:
        parser.error("--mesh requires a named --case to preserve the core ablations")
    out = Path(args.out)
    report = json.loads((out / "ablation_report.json").read_text())
    voxel = report["voxel"]
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.source).resolve()), use_scripts=False)
    source, _ = source_bvh()
    observed = np.load(out / "observed_gt_samples.npy")
    sharp_flags = None
    observed_sharp = None
    if args.source_flags:
        dihedral = np.load(args.source_flags)
        sharp_flags = dihedral > args.source_flags_threshold
        _, observed_index = distances(source, observed)
        observed_sharp = sharp_flags[observed_index]
    results = {"protocol": "50k fixed-seed area-weighted surface samples -> exact source triangles; completeness uses 50k observed GT depth points -> mesh triangles; not an official benchmark",
               "voxel": voxel, "cases": {}}
    if args.case and (out / "surface_metrics.json").is_file():
        results = json.loads((out / "surface_metrics.json").read_text())
    for name in ([args.case] if args.case else report["cases"]):
        start = time.time()
        if args.mesh:
            mesh_path = Path(args.mesh)
            if mesh_path.suffix == ".obj":
                v, f = [], []
                for line in mesh_path.read_text().splitlines():
                    if line.startswith("v "):
                        v.append([float(x) for x in line.split()[1:4]])
                    elif line.startswith("f "):
                        f.append([int(x.split("/")[0])-1 for x in line.split()[1:4]])
                vertices, faces = np.asarray(v), np.asarray(f)
            else:
                vertices, faces, _ = read_ply(mesh_path)
            if args.mesh_is_omega:
                vertices = report["scale"] * vertices @ np.asarray(report["rotation"]).T + report["translation"]
        else:
            vertices, faces, _ = read_ply(out / name / "mesh.ply")
        mesh = BVHTree.FromPolygons(vertices.tolist(), faces.tolist(), all_triangles=True)
        if args.mesh:
            tri = vertices[faces]
            area = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1)
            rng = np.random.default_rng(20261102)
            selected = rng.choice(len(faces), 50000, p=area/area.sum())
            a, b = np.sqrt(rng.random(50000)), rng.random(50000)
            bary = np.stack([1-a, a*(1-b), a*b], axis=1)
            points = (tri[selected]*bary[...,None]).sum(axis=1)
        else:
            points = np.load(out / name / "surface_samples.npz")["points"]
        accuracy, accuracy_index = distances(source, points)
        completeness, _ = distances(mesh, observed)
        normal_angles = []
        source_hits = []
        for p in points:
            loc, sn, _, _ = source.find_nearest(tuple(p))
            _, mn, _, _ = mesh.find_nearest(tuple(p))
            source_hits.append(loc)
            normal_angles.append(np.degrees(np.arccos(np.clip(abs(sn.dot(mn)), 0, 1))))
        source_hits = np.asarray(source_hits)
        scores = {}
        for multiple in (1, 2):
            precision = float((accuracy <= multiple*voxel).mean())
            recall = float((completeness <= multiple*voxel).mean())
            scores[f"fscore_{multiple}_voxel"] = {"precision": precision, "observed_recall": recall,
                                                  "fscore": 2*precision*recall/max(precision+recall, 1e-12)}
        stats = {"accuracy": summarize(accuracy, voxel), "observed_completeness": summarize(completeness, voxel),
                 "symmetric_mean_distance": float((accuracy.mean()+completeness.mean())/2),
                 "outlier_rate_accuracy_gt2vox": float((accuracy > 2*voxel).mean()),
                 "outlier_rate_accuracy_gt4vox": float((accuracy > 4*voxel).mean()),
                 "outlier_rate_completeness_gt2vox": float((completeness > 2*voxel).mean()),
                 "unsigned_normal_mean_deg": float(np.mean(normal_angles)),
                 "unsigned_normal_median_deg": float(np.median(normal_angles)), **scores}
        # Orientation split: accuracy by azimuth sector of the nearest source point,
        # completeness by azimuth sector of the observed GT point (world frame).
        az_acc = np.degrees(np.arctan2(source_hits[:, 1], source_hits[:, 0]))
        az_obs = np.degrees(np.arctan2(observed[:, 1], observed[:, 0]))
        sectors = [((-180 + 45 * i), (-180 + 45 * (i + 1))) for i in range(8)]
        stats["orientation_split"] = {
            "accuracy_mean_in_voxels_by_azimuth": {
                f"{lo}..{hi}": float(accuracy[(az_acc >= lo) & (az_acc < hi)].mean() / voxel)
                if ((az_acc >= lo) & (az_acc < hi)).any() else None for lo, hi in sectors},
            "completeness_mean_in_voxels_by_azimuth": {
                f"{lo}..{hi}": float(completeness[(az_obs >= lo) & (az_obs < hi)].mean() / voxel)
                if ((az_obs >= lo) & (az_obs < hi)).any() else None for lo, hi in sectors}}
        if sharp_flags is not None:
            acc_sharp = sharp_flags[accuracy_index]
            stats["structure_split"] = {
                "threshold_deg": float(args.source_flags_threshold),
                "sharp_fraction_of_source": float(sharp_flags.mean()),
                "accuracy_sharp": summarize(accuracy[acc_sharp], voxel),
                "accuracy_flat": summarize(accuracy[~acc_sharp], voxel),
                "completeness_sharp": summarize(completeness[observed_sharp], voxel),
                "completeness_flat": summarize(completeness[~observed_sharp], voxel),
                "precision_sharp@1vox": float((accuracy[acc_sharp] <= voxel).mean()),
                "recall_sharp@1vox": float((completeness[observed_sharp] <= voxel).mean()),
            }
        stats["seconds"] = round(time.time()-start, 2)
        results["cases"][name] = stats
        print(name, json.dumps(stats), flush=True)
        (out / "surface_metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


