import json
import os

# anchor16 lives under deliverables/lego_p0 (its case key differs)
PATHS = {
    'set8': (r'G:\GitHub\3D-Tex\outputs\capture_set8', None),
    'set16': (r'G:\GitHub\3D-Tex\outputs\capture_set16', None),
    'set32': (r'G:\GitHub\3D-Tex\outputs\capture_set32', None),
    'set64': (r'G:\GitHub\3D-Tex\outputs\capture_set64', None),
    'anchor16': (r'G:\GitHub\3D-Tex\deliverables\lego_p0', 'pred_pose_pred_depth'),
    'dualring': (r'G:\GitHub\3D-Tex\outputs\capture_dualring', None),
    'res256': (r'G:\GitHub\3D-Tex\outputs\capture_res256', None),
    'res400': (r'G:\GitHub\3D-Tex\outputs\capture_res400', None),
    'res512': (r'G:\GitHub\3D-Tex\outputs\capture_res512', None),
}
rows = []
for s, (base, want_case) in PATHS.items():
    if not os.path.isdir(base):
        continue
    sm_path = os.path.join(base, 'surface_metrics.json')
    if os.path.isfile(sm_path):
        sm = json.load(open(sm_path))['cases']
        key = want_case or list(sm)[0]
        c = sm[key]
    else:
        # capture_eval-only outputs (no mesh surface metrics yet)
        c = {}
    ab = json.load(open(os.path.join(base, 'ablation_report.json')))
    dog = ab.get('depth_on_gt_grid') or {}
    legacy = ab.get('depth_on_fixed_gt_grid') or {}
    # Normalise to the capture-schema keys (anchor16 predates it).
    if dog:
        normal_mean = dog['normal_deg']['mean']
        flat_mean = dog['normal_deg_flat']['mean']
        edge_mean = dog['normal_deg_edge']['mean']
        low = dog.get('residual_lowfreq_rms')
        high = dog.get('residual_highfreq_rms')
    else:
        normal_mean = legacy.get('normal_mean_deg')
        flat_mean = legacy.get('flat_normal_mean_deg')
        edge_mean = None
        low = high = None
    ss = c.get('structure_split', {}) if c else {}
    ori = c.get('orientation_split', {}) if c else {}
    acc_by_az = ori.get('accuracy_mean_in_voxels_by_azimuth', {}) if ori else {}
    vals = [v for v in acc_by_az.values() if v is not None]
    rows.append(dict(
        s=s,
        F=(c.get('fscore_1_voxel') or {}).get('fscore') if c else None,
        P=(c.get('fscore_1_voxel') or {}).get('precision') if c else None,
        R=(c.get('fscore_1_voxel') or {}).get('observed_recall') if c else None,
        acc_vox=(c.get('accuracy') or {}).get('mean_in_voxels') if c else None,
        compl_vox=(c.get('observed_completeness') or {}).get('mean_in_voxels') if c else None,
        acc_sharp=(ss.get('accuracy_sharp') or {}).get('mean_in_voxels'),
        acc_flat=(ss.get('accuracy_flat') or {}).get('mean_in_voxels'),
        P_sharp=ss.get('precision_sharp@1vox'),
        R_sharp=ss.get('recall_sharp@1vox'),
        low=low,
        high=high,
        burr=ab.get('burr_rate'),
        hole_fill=(ab.get('structure') or {}).get('hole_fill_rate'),
        coverage=(ab.get('structure') or {}).get('coverage_rate'),
        normal=normal_mean,
        flat_n=flat_mean,
        edge_n=edge_mean,
        acc_gt2vox=c.get('outlier_rate_accuracy_gt2vox') if c else None,
        acc_gt4vox=c.get('outlier_rate_accuracy_gt4vox') if c else None,
        worst_az=max(vals) if vals else None,
        best_az=min(vals) if vals else None,
        abs_rel=(dog.get('abs_rel') or {}).get('mean') if dog else None,
    ))

hdr = ("set F P R acc compl sharp flat P_sh R_sh nrm flat_n edge_n "
       "low high burr hole_fill cover acc>2vox acc>4vox worst_az best_az")
print(hdr)
for r in rows:
    def f(x, fmt="{:.4f}"):
        return (fmt.format(x) if isinstance(x, float) else "—")
    print(r['s'], f(r['F']), f(r['P'], "{:.3f}"), f(r['R'], "{:.3f}"),
          f(r['acc_vox'], "{:.2f}"), f(r['compl_vox'], "{:.2f}"),
          f(r['acc_sharp'], "{:.2f}"), f(r['acc_flat'], "{:.2f}"),
          f(r['P_sharp'], "{:.3f}"), f(r['R_sharp'], "{:.3f}"),
          f(r['normal'], "{:.2f}"), f(r['flat_n'], "{:.2f}"), f(r['edge_n'], "{:.2f}"),
          f(r['low'], "{:.4f}"), f(r['high'], "{:.4f}"),
          f(r['burr'], "{:.4f}"), f(r['hole_fill'], "{:.3f}"), f(r['coverage'], "{:.3f}"),
          f(r['acc_gt2vox'], "{:.3f}"), f(r['acc_gt4vox'], "{:.3f}"),
          f(r['worst_az'], "{:.2f}"), f(r['best_az'], "{:.2f}"))
