#!/usr/bin/env python3
"""打印 T1 summary.json 里各条目的 aggregate（含 std），供报告引用。"""
import json
import sys

for path in sys.argv[1:]:
    d = json.load(open(path))
    print(f"# {path}")
    for k, v in d.items():
        agg = v.get("aggregate")
        if agg is None:
            continue
        print(f"{k}: {json.dumps(agg, ensure_ascii=False)}")
        print(f"{k} seam: {json.dumps(v.get('seam', {}), ensure_ascii=False)}")
        print(f"{k} material: {json.dumps(v.get('material', {}), ensure_ascii=False)}")
        print(f"{k} zero_vis_texels_share: {v.get('zero_vis_texels_share')}")
