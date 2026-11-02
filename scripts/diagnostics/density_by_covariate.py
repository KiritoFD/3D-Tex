"""Compare per-covariate-quartile abs_rel between set8 (sparse) and set64 (dense).

If densification helps most in low-coverage / low-texture / grazing pixels,
the failure there is information-driven. If the residual persists at the
same level in high-view / high-texture / frontal pixels, the failure is
network-prior-driven.
"""
import json
import numpy as np

COVS = ['views', 'parallax', 'incidence', 'img_grad', 'dist_to_dis',
        'curv', 'dist_to_sil']

rows = {}
for s in ['set8', 'set64']:
    d = np.load(rf'G:\GitHub\3D-Tex\outputs\pixel_attr\{s}\pixel_attr.npz')
    rows[s] = {k: d[k] for k in d.files}

print(f"{'covariate':>15} {'quartile':>8} {'set8':>8} {'set64':>8} {'gain':>8}")
print('-' * 55)
for c in COVS:
    x8 = rows['set8'][c]
    v8 = rows['set8']['valid']
    a8 = rows['set8']['abs_rel'][v8]
    a64 = rows['set64']['abs_rel'][rows['set64']['valid']]
    x8_v = x8[v8]
    x64 = rows['set64'][c][rows['set64']['valid']]
    qs = np.percentile(x8_v, [0, 25, 50, 75, 100])
    for qi, (lo, hi) in enumerate(zip(qs[:-1], qs[1:])):
        m8 = (x8_v >= lo) & (x8_v <= hi)
        m64 = (x64 >= lo) & (x64 <= hi)
        if m8.sum() < 50 or m64.sum() < 50:
            continue
        mean8 = a8[m8].mean()
        mean64 = a64[m64].mean()
        gain = (mean64 - mean8) / mean8 * 100
        print(f"{c:>15} Q{qi + 1:<7} {mean8:>8.4f} {mean64:>8.4f} {gain:>+7.1f}%")
