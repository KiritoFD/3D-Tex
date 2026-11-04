#!/usr/bin/env bash
# Phase-3 G1/G2 Lego runs: B0-coloured M2 lattice baselines, sequential on one GPU.
set -euo pipefail
cd /root/3dtex
PY=/opt/conda/envs/3dtex/bin/python
COMMON=(--mesh outputs/geometry_pruning/eval_carve_v35c30_conf/carve_v35c30/mesh.ply
        --omega data/scenes/nerf_lego16/geometry/omega
        --gt data/scenes/nerf_lego_gt16/geometry/gt
        --camera-mode diagnostic --resolution 256 --iters 300 --lr 0.05
        --voxel-size 0.01150144350298549 --lambda-sil 0.3 --lambda-fs 0.5
        --lambda-reg 0.1 --seed 20261103)

run() {
  local name=$1; shift
  echo "[p3] start $name $(date -Is)"
  $PY /tmp/peak_sample.py $PY scripts/m2_anchor_opt.py "${COMMON[@]}" --out "outputs/72h/phase3/$name" "$@" \
      > "outputs/72h/logs/p3_$name.log" 2>&1
  echo "[p3] done $name $(date -Is)"
}

run lego_g1_l4 --deformation lattice --lattice-spacing-voxels 4 --lambda-prior 0
run lego_g1_l4_rep2 --deformation lattice --lattice-spacing-voxels 4 --lambda-prior 0
run lego_g1_l8 --deformation lattice --lattice-spacing-voxels 8 --lambda-prior 0
run lego_g2_l4 --deformation lattice --lattice-spacing-voxels 4 \
    --lambda-prior 0.05 --tolerance-voxels 2 --lambda-normal 0.2
run lego_g2_l4_rep2 --deformation lattice --lattice-spacing-voxels 4 \
    --lambda-prior 0.05 --tolerance-voxels 2 --lambda-normal 0.2
echo "[p3] all done $(date -Is)"
