#!/bin/bash
# Tex-T1 §9.3 材质轴：非平凡 MRA 输入（image / preset 常数）× proxy × bf16 × seed 42
# 口径 = proxy bf16 主行（2048 + remesh + dec 500000）；判定口径见 TEX_T1_SUPRA.md §9.3
# （2026-11-12 预注册）：roughness_pred_minus_input 与 roughness_std 两个证据合并归类。
set -uo pipefail
ROOT=/root/3dtex
OUT=$ROOT/outputs/tex_t1_supra_20261105
MIMG=$ROOT/outputs/tex_t1_mra_image_20261112
MPRE=$ROOT/outputs/tex_t1_mra_preset_r025_20261112
SPY=/opt/conda/envs/supra-tex/bin/python
TPY=/opt/conda/envs/3dtex/bin/python
SRC=$ROOT/third_party/supra_tex
LOG=$MIMG/logs/mra_runs.log
mkdir -p "$MIMG/logs" "$MIMG/glb_bf16" "$MPRE/glb_bf16" "$MPRE/logs"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---- GPU 共享锁：短持锁轮询 ----
mkdir -p "$ROOT/.locks"
NEED_FREE=${FREE_THRESHOLD_MB:-10500}
WAIT_WINDOW=${WAIT_WINDOW_S:-7200}
deadline=$(( $(date +%s) + WAIT_WINDOW ))
say "poll for GPU (need free>=${NEED_FREE}MiB, window=${WAIT_WINDOW}s)"
exec 9>"$ROOT/.locks/gpu4090.lock"
while :; do
  if ! flock -n 9; then
    flock -u 9 2>/dev/null || true
    [ "$(date +%s)" -ge "$deadline" ] && { say "giveup: lock busy"; exit 2; }
    sleep $((35 + RANDOM % 30)); continue
  fi
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -ge "$NEED_FREE" ]; then say "acquired, free=${free}MiB"; break; fi
  say "free=${free}MiB < ${NEED_FREE}MiB, release and retry"
  flock -u 9 2>/dev/null || true
  [ "$(date +%s)" -ge "$deadline" ] && { say "giveup: vram free=${free}MiB"; exit 2; }
  sleep $((35 + RANDOM % 30))
done
BASE_VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
say "baseline used=${BASE_VRAM}MiB"
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" >> "$MIMG/logs/vram_mra_all.txt"; sleep 2; done ) &
MON=$!
trap 'kill $MON 2>/dev/null; flock -u 9' EXIT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_one() {  # $1=variant root  $2=suffix (mraimage / mrapreset)
  local vroot="$1" sfx="$2"
  local name="supra_proxy_${sfx}_v0_s42_bf16"
  local glb="$vroot/glb_bf16/${name}.glb"
  if [ -f "$glb" ]; then say "skip $name (exists)"; return 0; fi
  say "run $name"
  local t0=$(date +%s)
  ( cd "$SRC" && SUPRATEX_ABI_IGNORE=dtype SUPRATEX_FORCE_BF16=1 "$SPY" texturing.py \
      --model-config "$ROOT/models/supra_tex/completion_bf16/config.json" \
      --checkpoint "$ROOT/models/supra_tex/completion/denoiser_step0030000.safetensors" \
      --trellis-root "$ROOT/models/trellis2" \
      --input-dir "$vroot/inputs/proxy_v0" --output "$glb" \
      --attention sdpa --low-vram --seed 42 --texture-size 2048 --texture-format png ) \
    >> "$MIMG/logs/${name}.log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  local peak=$(awk -v a="$t0" -v b="$t1" '$1>=a && $1<=b { if ($2+0>m) m=$2+0 } END { print m+0 }' "$MIMG/logs/vram_mra_all.txt")
  if [ $rc -eq 0 ] && [ -f "$glb" ]; then
    say "OK $name elapsed=$((t1-t0))s peak_all=${peak}MiB peak_delta=$((peak-BASE_VRAM))MiB"
  else
    local reason=$(grep -o "CUDA out of memory" "$MIMG/logs/${name}.log" 2>/dev/null | head -1)
    say "FAIL $name rc=$rc elapsed=$((t1-t0))s peak_all=${peak}MiB reason=${reason:-unknown}"
  fi
}

run_one "$MIMG" mraimage
run_one "$MPRE" mrapreset

# ---- 材质对照评测：--mra-ref 显式指向各自输入包的 mra.png ----
if [ -f "$MIMG/glb_bf16/supra_proxy_mraimage_v0_s42_bf16.glb" ] && \
   [ -f "$MPRE/glb_bf16/supra_proxy_mrapreset_v0_s42_bf16.glb" ]; then
  say "eval material-axis"
  "$TPY" "$ROOT/scripts/experiments/tex_t1_eval.py" \
    --glb "supra_proxy_mraimage_v0_s42_bf16=$MIMG/glb_bf16/supra_proxy_mraimage_v0_s42_bf16.glb" \
    --glb "supra_proxy_mrapreset_v0_s42_bf16=$MPRE/glb_bf16/supra_proxy_mrapreset_v0_s42_bf16.glb" \
    --mra-ref "supra_proxy_mraimage_v0_s42_bf16=$MIMG/inputs/proxy_v0/mra.png" \
    --mra-ref "supra_proxy_mrapreset_v0_s42_bf16=$MPRE/inputs/proxy_v0/mra.png" \
    --out "$MIMG/eval_mra" --save-renders >> "$MIMG/logs/eval_mra.log" 2>&1
  say "eval rc=$? -> $MIMG/eval_mra/summary.json"
else
  say "eval skipped: missing GLB(s)"
fi
say "DONE"
