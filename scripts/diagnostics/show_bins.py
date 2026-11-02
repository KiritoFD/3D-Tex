import json
import sys

d = json.load(open(r'G:\GitHub\3D-Tex\outputs\pixel_attr\set32\covariate_corr.json'))
for c in ['views', 'parallax', 'incidence', 'img_grad', 'dist_to_dis',
          'curv', 'dist_to_sil']:
    bins = d[c]['bins']
    means = [b['abs_rel'] for b in bins]
    print(f"{c:>14}: q1={means[0]:.4f} q2={means[1]:.4f} q3={means[2]:.4f} "
          f"q4={means[3]:.4f} rho={d[c]['spearman_rho']:+.3f}")
