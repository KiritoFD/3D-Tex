#!/bin/bash
# Phase-4 A2: training-frustum + predicted-occlusion visibility masks, then masked re-eval.
set -u
ROOT=/root/3dtex
PY=/opt/conda/envs/3dtex/bin/python
cd "$ROOT" || exit 1
LOG=$ROOT/outputs/72h/logs
mkdir -p "$ROOT/outputs/72h/phase4/a2/masks"

for scene in eth3d_kicker eth3d_pipes eth3d_relief; do
  p2=$ROOT/outputs/72h/phase2/$scene
  a1=$ROOT/outputs/72h/phase4/a1/$scene
  arms=(--arm "b0=$p2/b0/mesh/mesh_init.ply" --arm "b1=$p2/b1/mesh/mesh_init.ply")
  if [ -f "$a1/c1/mesh/mesh_init.ply" ]; then
    arms+=(--arm "a1_c1=$a1/c1/mesh/mesh_init.ply" --arm "a1_c2=$a1/c2/mesh/mesh_init.ply")
  fi
  "$PY" scripts/diagnostics/visibility_eval.py --scene "$scene" \
    --scene-dir "$ROOT/data/scenes/$scene" --phase2 "$p2" \
    "${arms[@]}" \
    --out "$ROOT/outputs/72h/phase4/a2/${scene}_visibility.json" \
    --mask-out "$ROOT/outputs/72h/phase4/a2/masks/${scene}_visible.npy" \
    >> "$LOG/p4_a2_${scene}.log" 2>&1
  echo "a2 $scene done $(date +%H:%M:%S)" >> "$LOG/p4_a2_all.log"
done
echo a2-all-done >> "$LOG/p4_a2_all.log"
