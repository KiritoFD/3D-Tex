#!/bin/bash
# Phase-4 A1: GT-free background-shell suppression (confidence + free-space carving).
# Two fixed configurations per scene (no parameter sweep), frozen P0 grid reused.
#   1) carve kicker/pipes -> TSDF on the frozen B0 grid -> equal-protocol metrics
#   2) Lego regression guard through the frozen P0 capture pipeline
set -u
ROOT=/root/3dtex
PY=/opt/conda/envs/3dtex/bin/python
BLENDER=${BLENDER:-/usr/local/bin/blender}
cd "$ROOT" || exit 1
LOG=$ROOT/outputs/72h/logs

declare -A FAR=( [eth3d_kicker]=4.3022 [eth3d_pipes]=5.1896 )
declare -A VOX=( [eth3d_kicker]=0.018257796759500712 [eth3d_pipes]=0.028879645450190592 )

carve_params () {
  case "$1" in
    c1) CONF=0.2; VOTE=0.7; CONFIRM=0.05; REL=0.010 ;;
    c2) CONF=0.5; VOTE=0.6; CONFIRM=0.03; REL=0.015 ;;
  esac
}

run_eth3d () {
  local scene=$1 cfg=$2
  local b0=$ROOT/outputs/72h/phase2/$scene/b0
  local out=$ROOT/outputs/72h/phase4/a1/$scene/$cfg
  local geom=$out/geometry/omega
  carve_params "$cfg"
  mkdir -p "$geom"
  "$PY" scripts/free_space_carving.py --omega "$b0/geometry/omega" --out "$geom" \
    --conf-quantile "$CONF" --tau-voxels 3 --tau-rel "$REL" --voxel "${VOX[$scene]}" \
    --vote-frac "$VOTE" --confirm-max "$CONFIRM" --min-views 3 \
    >> "$LOG/p4_a1_${scene}.log" 2>&1
  "$PY" scripts/tsdf_mesh.py --scene "$scene" --filtered "$geom" \
    --roi-geometry "$b0/geometry/omega" \
    --depth-min 0 --depth-trunc "${FAR[$scene]}" \
    --weight-mode confidence --sample-mode nearest \
    --out "$out/mesh" >> "$LOG/p4_a1_${scene}.log" 2>&1
  echo "$scene/$cfg done $(date +%H:%M:%S)" >> "$LOG/p4_a1_all.log"
}

report_eth3d () {
  local scene=$1
  local p2=$ROOT/outputs/72h/phase2/$scene
  local dir=$ROOT/outputs/72h/phase4/a1/$scene
  "$PY" scripts/diagnostics/eth3d_protocol_calibration.py --scene "$scene" \
    --scene-dir "$ROOT/data/scenes/$scene" --phase2 "$p2" \
    --out "$dir/eval/protocol_$scene.json" \
    --arm "b0=$p2/b0/mesh/mesh_init.ply" \
    --arm "b1=$p2/b1/mesh/mesh_init.ply" \
    --arm "a1_c1=$dir/c1/mesh/mesh_init.ply" \
    --arm "a1_c2=$dir/c2/mesh/mesh_init.ply" \
    >> "$LOG/p4_a1_eval_${scene}.log" 2>&1
  echo "eval $scene done $(date +%H:%M:%S)" >> "$LOG/p4_a1_all.log"
}

run_lego () {
  local cfg=$1
  local out=$ROOT/outputs/72h/phase4/a1/lego_reg/$cfg
  local geom=$out/geometry/omega
  local capture=$out/capture
  carve_params "$cfg"
  mkdir -p "$geom" "$capture/geometry"
  "$PY" scripts/free_space_carving.py \
    --omega "$ROOT/outputs/72h/phase2/lego_b0/geometry/omega" --out "$geom" \
    --conf-quantile "$CONF" --tau-voxels 3 --tau-rel "$REL" --voxel 0.01150144350298549 \
    --vote-frac "$VOTE" --confirm-max "$CONFIRM" --min-views 3 \
    >> "$LOG/p4_a1_lego.log" 2>&1
  ln -sfn "$geom" "$capture/geometry/omega"
  ln -sfn "$ROOT/data/scenes/nerf_lego_gt16/geometry/gt" "$capture/geometry/gt"
  "$PY" scripts/capture/capture_eval.py --capture "$capture" --case "$cfg" --out "$out/p0" \
    --weight-mode confidence --fusion-sample nearest \
    >> "$LOG/p4_a1_lego.log" 2>&1
  "$PY" scripts/mesh_opt/evaluate.py --out "$out/p0" --meshes "$cfg=$out/p0/$cfg/mesh.ply" \
    --blender "$BLENDER" >> "$LOG/p4_a1_lego.log" 2>&1
  echo "lego/$cfg done $(date +%H:%M:%S)" >> "$LOG/p4_a1_all.log"
}

case "${1:-all}" in
  all)
    for scene in eth3d_kicker eth3d_pipes; do
      for cfg in c1 c2; do run_eth3d "$scene" "$cfg"; done
      report_eth3d "$scene"
    done
    for cfg in c1 c2; do run_lego "$cfg"; done
    echo all-done >> "$LOG/p4_a1_all.log"
    ;;
  eth3d)
    for scene in eth3d_kicker eth3d_pipes; do
      for cfg in c1 c2; do run_eth3d "$scene" "$cfg"; done
      report_eth3d "$scene"
    done
    ;;
  lego)
    for cfg in c1 c2; do run_lego "$cfg"; done
    ;;
  eval)
    for scene in eth3d_kicker eth3d_pipes; do report_eth3d "$scene"; done
    ;;
esac
