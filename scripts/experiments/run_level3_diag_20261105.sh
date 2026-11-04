#!/bin/bash
# Level 3 attribution runs: init extraction, finer grid, stronger L_dev.
set -u
cd /root/3dtex || exit 1
OUT=outputs/step_baseline_20261105
LOG=$OUT/train_diag.log
CORR=$OUT/correspondence.npz
PY=/opt/conda/envs/3dtex/bin/python
: > "$LOG"
nvidia-smi --query-gpu=memory.used --format=csv >> "$LOG"

run() {
  name=$1; shift
  echo "=== $name start $(date -Is) ===" >> "$LOG"
  PYTHONPATH=scripts "$PY" scripts/experiments/step3_flexicubes.py \
    --out "$OUT" --run "$name" "$@" >> "$LOG" 2>&1
  echo "=== $name exit $? $(date -Is) ===" >> "$LOG"
  nvidia-smi --query-gpu=memory.used --format=csv >> "$LOG"
}

# Extraction only: one step, geometry supervision disabled.
run init_sA --seed 20261105 --iters 1 --sil-weight 0 --corr-weight 0
# Same supervision, finer grid (requested 256 was skipped in the main runs due to memory caution).
run g256_sA --seed 20261105 --grid-res 256 --iters 100 --sil-weight 0.5 --corr-weight 0.02 --corr-file "$CORR"
# Stronger FlexiCubes L_dev (10x) at the main grid resolution.
run ldev_sA --seed 20261105 --iters 100 --sil-weight 0.5 --corr-weight 0.02 --corr-file "$CORR" --l-dev-weight 1.0
echo "=== all done $(date -Is) ===" >> "$LOG"
