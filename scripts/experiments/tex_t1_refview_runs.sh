#!/bin/bash
# Tex-T1 §9.4 参考视角敏感性：proxy bf16 seed 42 × views {4,8,12}（v0 复用既有 bf16 主行）
# + 同视角 T0 对照（--cam-indices）+ 逐条目参考视角评测 + 跨参考视角渲染 MAD（CPU）
# 判定口径见 TEX_T1_SUPRA.md §9.4（2026-11-12 预注册）：psnr_unseen 视角间极差。
set -uo pipefail
ROOT=/root/3dtex
OUT=$ROOT/outputs/tex_t1_supra_20261105
RV=$ROOT/outputs/tex_t1_refview_20261112
SPY=/opt/conda/envs/supra-tex/bin/python
TPY=/opt/conda/envs/3dtex/bin/python
SRC=$ROOT/third_party/supra_tex
SCENE=$ROOT/data/scenes/nerf_lego_gt16
LOG=$RV/logs/refview_runs.log
mkdir -p "$RV/logs"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---- GPU 共享锁：短持锁轮询（推理与 T0 渲染都占 GPU）----
mkdir -p "$ROOT/.locks"
NEED_FREE=${FREE_THRESHOLD_MB:-10500}
WAIT_WINDOW=${WAIT_WINDOW_S:-10800}
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
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" >> "$RV/logs/vram_refview_all.txt"; sleep 2; done ) &
MON=$!
trap 'kill $MON 2>/dev/null; flock -u 9' EXIT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- 1) SUPRA：参考视角 4/8/12（bf16、2048 + remesh + dec 500000，与主行同规格）----
for v in 4 8 12; do
  name="supra_proxy_v${v}_s42_bf16"
  glb="$OUT/glb_bf16/${name}.glb"
  if [ -f "$glb" ]; then say "skip $name (exists)"; continue; fi
  say "run $name"
  t0=$(date +%s)
  ( cd "$SRC" && SUPRATEX_ABI_IGNORE=dtype SUPRATEX_FORCE_BF16=1 "$SPY" texturing.py \
      --model-config "$ROOT/models/supra_tex/completion_bf16/config.json" \
      --checkpoint "$ROOT/models/supra_tex/completion/denoiser_step0030000.safetensors" \
      --trellis-root "$ROOT/models/trellis2" \
      --input-dir "$OUT/inputs/proxy_v${v}" --output "$glb" \
      --attention sdpa --low-vram --seed 42 --texture-size 2048 --texture-format png ) \
    >> "$RV/logs/${name}.log" 2>&1
  rc=$?
  t1=$(date +%s)
  peak=$(awk -v a="$t0" -v b="$t1" '$1>=a && $1<=b { if ($2+0>m) m=$2+0 } END { print m+0 }' "$RV/logs/vram_refview_all.txt")
  if [ $rc -eq 0 ] && [ -f "$glb" ]; then
    say "OK $name elapsed=$((t1-t0))s peak_all=${peak}MiB peak_delta=$((peak-BASE_VRAM))MiB"
  else
    reason=$(grep -o "CUDA out of memory" "$RV/logs/${name}.log" 2>/dev/null | head -1)
    say "FAIL $name rc=$rc elapsed=$((t1-t0))s peak_all=${peak}MiB reason=${reason:-unknown}"
  fi
done

# ---- 2) T0 对照：单参考相机 = 对应视角（--cam-indices 与 --max-cams 互斥）----
for v in 4 8 12; do
  td="$RV/t0_v$v"
  if [ -f "$td/proxy_atlas2048_weighted_mean.png" ]; then say "skip T0 v$v (exists)"; continue; fi
  mkdir -p "$td"
  say "T0 v$v"
  "$TPY" "$ROOT/scripts/experiments/texture_fixed_mesh.py" \
    --meshes proxy --atlas 2048 --cam-indices "$v" --variants weighted_mean,best_view \
    --out "$td" >> "$RV/logs/t0_v$v.log" 2>&1
  say "T0 v$v rc=$?"
done

# ---- 3) 逐视角评测：SUPRA 与同视角 T0 同一口径（seen 定义用该参考相机）----
ok_eval=1
for v in 4 8 12; do
  glb="$OUT/glb_bf16/supra_proxy_v${v}_s42_bf16.glb"
  [ -f "$glb" ] || { say "eval v$v skipped (no GLB)"; ok_eval=0; continue; }
  say "eval v$v"
  "$TPY" "$ROOT/scripts/experiments/tex_t1_eval.py" \
    --glb t0:proxy --ref-view "t0:proxy=$v" --t0-dir "$RV/t0_v$v" \
    --glb "supra_proxy_v${v}_s42_bf16=$glb" \
    --out "$RV/eval_v$v" --save-renders >> "$RV/logs/eval_v$v.log" 2>&1
  say "eval v$v rc=$?"
done

# ---- 4) 跨参考视角渲染 MAD（CPU，{0,4} {0,8} {0,12}）----
if [ $ok_eval -eq 1 ]; then
  say "render MAD"
  "$TPY" "$ROOT/scripts/experiments/tex_t1_refview_render_mad.py" \
    --ref-render-dir "$OUT/eval_final/renders" --ref-name "supra_proxy_v0_s42_bf16" \
    --eval-root "$RV" --pairs "0:4,0:8,0:12" \
    --out "$RV/render_mad.json" >> "$RV/logs/render_mad.log" 2>&1
  say "render MAD rc=$?"
fi
say "DONE"
