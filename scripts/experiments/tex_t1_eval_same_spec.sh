#!/usr/bin/env bash
# Tex-T1：同规格（1024 + 不重网格化 + bf16）proxy / vggt 的四轴评测
# 复用 eval_final 的修正后 axis/UV 口径（tex_t1_eval.py），只写新子目录，不覆盖旧产物。
set -u
ROOT=/root/3dtex
OUT=${OUT_ROOT:-$ROOT/outputs/tex_t1_supra_20261105}
TPY=/opt/conda/envs/3dtex/bin/python
GLB_DIR=${GLB_DIR:-$OUT/glb_same_spec_1024}
EVAL_DIR=${EVAL_DIR:-$OUT/eval_same_spec_1024}
FREE_THRESHOLD=${FREE_THRESHOLD:-3500}
WAIT_WINDOW=${WAIT_WINDOW_S:-1800}
LOG=$OUT/logs/eval_same_spec_1024.log
mkdir -p "$OUT/logs" "$EVAL_DIR"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

deadline=$(( $(date +%s) + WAIT_WINDOW ))
exec 9>"$ROOT/.locks/gpu4090.lock"
while :; do
  if ! flock -n 9; then
    flock -u 9 2>/dev/null || true
    [ "$(date +%s)" -ge "$deadline" ] && { say "giveup: lock busy"; exit 2; }
    say "lock busy, retry in 45s"; sleep 45; continue
  fi
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -ge "$FREE_THRESHOLD" ]; then say "acquired, free=${free}MiB"; break; fi
  say "free=${free}MiB < ${FREE_THRESHOLD}MiB, release and retry"
  flock -u 9 2>/dev/null || true
  [ "$(date +%s)" -ge "$deadline" ] && { say "giveup: vram free=${free}MiB"; exit 2; }
  sleep 45
done

"$TPY" "$ROOT/scripts/experiments/tex_t1_eval.py" \
  --glb t0:proxy --glb t0:vggt \
  --glb "supra_proxy_v0_s42_bf16_no_remesh=$GLB_DIR/supra_proxy_v0_s42_bf16_no_remesh.glb" \
  --glb "supra_vggt_v0_s42_bf16_no_remesh=$OUT/glb_bf16/supra_vggt_v0_s42_bf16_no_remesh.glb" \
  --out "$EVAL_DIR" --save-renders >> "$LOG" 2>&1
rc=$?
say "eval rc=$rc -> $EVAL_DIR/summary.json"
flock -u 9 2>/dev/null || true
exit $rc
