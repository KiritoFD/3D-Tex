#!/bin/bash
# Level 3 (FlexiCubes) controlled runs, 2026-11-05.
set -u
cd /root/3dtex || exit 1
OUT=outputs/step_baseline_20261105
LOG=$OUT/train.log
CORR=$OUT/correspondence.npz
PY=/opt/conda/envs/3dtex/bin/python
mkdir -p "$OUT"
: > "$LOG"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv >> "$LOG"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv >> "$LOG"

run() {
  name=$1; shift
  echo "=== $name start $(date -Is) ===" >> "$LOG"
  PYTHONPATH=scripts "$PY" scripts/experiments/step3_flexicubes.py \
    --out "$OUT" --run "$name" "$@" >> "$LOG" 2>&1
  echo "=== $name exit $? $(date -Is) ===" >> "$LOG"
  nvidia-smi --query-gpu=memory.used --format=csv >> "$LOG"
}

run full_sA --seed 20261105 --sil-weight 0.5 --corr-weight 0.02 --corr-file "$CORR"
run full_sB --seed 20261106 --sil-weight 0.5 --corr-weight 0.02 --corr-file "$CORR"
run nocorr_sA --seed 20261105 --sil-weight 0.5 --corr-weight 0
run nosil_sA --seed 20261105 --sil-weight 0 --corr-weight 0.02 --corr-file "$CORR"
echo "=== all done $(date -Is) ===" >> "$LOG"
