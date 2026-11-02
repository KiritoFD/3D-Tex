import json
import os

sets = ['set8', 'set16', 'set32', 'set64', 'dualring']
rows = []
for s in sets:
    p = rf'G:\GitHub\3D-Tex\outputs\pixel_attr\{s}\region_summary.json'
    if not os.path.isfile(p):
        continue
    d = json.load(open(p))
    row = {'set': s}
    for region, stats in d.items():
        if stats.get('count'):
            row[region] = stats['abs_rel_mean']
    rows.append(row)

regions = ['flat', 'weak_texture', 'high_curvature', 'thin_sharp', 'edge_band', 'highlight']
print(f"{'set':>9}  " + "  ".join(f"{r:>14}" for r in regions))
for r in rows:
    print(f"{r['set']:>9}  " + "  ".join(f"{r.get(reg, 0):>14.4f}" for reg in regions))

# Density slope: set8->set64 (uniform ring density axis)
print("\n== density benefit (set8 -> set64) ==")
low = next(r for r in rows if r['set'] == 'set8')
high = next(r for r in rows if r['set'] == 'set64')
dual = next(r for r in rows if r['set'] == 'dualring')
for reg in regions:
    if reg in low and reg in high:
        delta = high[reg] - low[reg]
        pct = 100 * delta / low[reg]
        print(f"  {reg:>14}: {low[reg]:.4f} -> {high[reg]:.4f}  ({pct:+.1f}%)")
print(f"\n  {'dualring vs set32 (elev-div vs same-N ring)'}:")
set32 = next(r for r in rows if r['set'] == 'set32')
for reg in regions:
    if reg in dual and reg in set32:
        delta = dual[reg] - set32[reg]
        pct = 100 * delta / set32[reg]
        print(f"  {reg:>14}: set32={set32[reg]:.4f} dual={dual[reg]:.4f}  ({pct:+.1f}%)")
