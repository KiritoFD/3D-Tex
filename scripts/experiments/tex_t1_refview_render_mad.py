#!/usr/bin/env python3
"""Tex-T1 §9.4：跨参考视角的输出一致性。

同一 heldout 视角下，比较"不同参考视角生成的 SUPRA 纹理"的渲染像素 MAD。
覆盖定义与 tex_t1_eval.py 的跨视角一致性一致：两侧渲染 RGB 之和 > 0 的像素。
这是描述性指标（判据只用 psnr_unseen 极差），不入预注册判定。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def load_pred(render_dir: Path, name: str, view: int) -> np.ndarray | None:
    p = render_dir / f"{name}_v{view:02d}_pred.png"
    if not p.is_file():
        return None
    return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-render-dir", required=True, help="参考视角 0 的评测渲染目录")
    ap.add_argument("--ref-name", required=True, help="参考视角 0 的条目名（v0）")
    ap.add_argument("--eval-root", required=True, help="含 eval_v{n}/renders 的根目录")
    ap.add_argument("--name-template", default="supra_proxy_v{v}_s42_bf16")
    ap.add_argument("--pairs", default="0:4,0:8,0:12", help="参考视角对 a:b")
    ap.add_argument("--heldout-views", default="16", help="heldout 视角数")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref_dir = Path(args.ref_render_dir)
    eval_root = Path(args.eval_root)
    pairs = [(int(a), int(b)) for p in args.pairs.split(",") if p.strip() for a, b in [p.split(":")]]
    n_heldout = int(args.heldout_views)

    def render_dir_for(v: int) -> Path:
        return ref_dir if v == 0 else eval_root / f"eval_v{v}" / "renders"

    def name_for(v: int) -> str:
        return args.name_template.format(v=v)

    out = {"pairs": [], "note": ("覆盖 = 两侧渲染 RGB 之和 > 0 的像素；"
                                 "与 tex_t1_eval.py 跨视角一致性的覆盖定义一致"),
           "ref_render_dir": str(ref_dir), "ref_name": args.ref_name}
    for a, b in pairs:
        diffs, n_common = [], 0
        missing = []
        for ci in range(n_heldout):
            ra = load_pred(render_dir_for(a), name_for(a), ci)
            rb = load_pred(render_dir_for(b), name_for(b), ci)
            if ra is None or rb is None or ra.shape != rb.shape:
                missing.append(ci)
                continue
            ok = (ra.sum(-1) > 0) & (rb.sum(-1) > 0)
            if ok.sum() == 0:
                continue
            diffs.append(np.abs(ra[ok] - rb[ok]).mean())
            n_common += int(ok.sum())
        rec = {
            "pair": [a, b], "views_compared": n_heldout - len(missing),
            "missing_views": missing, "common_covered_pixels": n_common,
            "mad": round(float(np.mean(diffs)), 5) if diffs else None,
            "mad_per_view": [round(float(d), 5) for d in diffs],
        }
        out["pairs"].append(rec)
        print(f"[pair {a}:{b}] mad={rec['mad']} views={rec['views_compared']} px={n_common}")

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
