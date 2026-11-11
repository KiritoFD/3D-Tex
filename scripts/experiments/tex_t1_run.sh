#!/bin/bash
# Tex-T1 v2：fp32 主臂排队 + 固定 Mesh SUPRA-Tex 推理 + T0 单视角对照 + 四轴评测
#
# v2（2026-11-12 准备，尚未运行）相对 v1 的变更——全部为排队/可观测性修复，不改实验口径：
#   1) 排队协议升级为「短持锁轮询」：拿锁 -> 查 free -> 不足立即放锁重试。
#      v1 的 fp32 四连败（2026-11-05 15:56/16:04/16:43/16:56）根因是 co-tenant 显存：
#      拿到锁时 baseline 已被占 17041MiB，OOM 时全卡 24070MiB。
#      证据：outputs/tex_t1_supra_20261105/logs/run.log、supra_vggt_v0_s42.log
#      （"Process 880404 has 14.96 GiB memory in use"）。见 TEX_T1_SUPRA.md §8。
#   2) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，降低碎片化 OOM 概率。
#   3) decimation target 显式化：默认沿用上游 inference_defaults 的 500000
#      （v1 未传参，实际生效值就是这个；v2 只是写出来，行为等价）。
#   4) 每 run 独立 status 文件（logs/fp32_${name}.status）与独立 VRAM 采样文件。
#
# 预注册的 fp32 主臂口径（TEX_T1_SUPRA.md §9.1）：2048 png + remesh + decimate 500k，
# proxy/vggt × seed 42/1234 共 4 个 run；评测见步骤 5。
set -uo pipefail
ROOT=/root/3dtex
OUT=${OUT_ROOT:-$ROOT/outputs/tex_t1_supra_20261105}
TEX=${TEX_SIZE:-2048}
DEC=${DEC_TARGET:-500000}
SPY=/opt/conda/envs/supra-tex/bin/python
TPY=/opt/conda/envs/3dtex/bin/python
SRC=$ROOT/third_party/supra_tex
LOG=$OUT/logs/run.log
mkdir -p "$OUT/logs" "$OUT/glb_fp32" "$OUT/inputs"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---- 0) CPU 侧前置检查（不占 GPU，任何一项失败直接退出）----
FP32_CONF=$ROOT/models/supra_tex/completion/config.json
FP32_CKPT=$ROOT/models/supra_tex/completion/denoiser_step0030000.safetensors
FP32_NORM=$ROOT/models/supra_tex/completion/partial_normalization.json
for f in "$FP32_CONF" "$FP32_CKPT" "$FP32_NORM"; do
  if [ ! -f "$f" ]; then say "FATAL: missing $f"; exit 3; fi
done
VAE_STEMS="shape_enc_next_dc_f16c32_fp16 tex_enc_next_dc_f16c32_fp16 shape_dec_next_dc_f16c32_fp16 tex_dec_next_dc_f16c32_fp16"
for stem in $VAE_STEMS; do
  if [ ! -f "$ROOT/models/trellis2/ckpts/${stem}.safetensors" ]; then
    say "FATAL: missing trellis2 ckpt ${stem}.safetensors"; exit 3
  fi
done
say "weights ok: completion fp32 ckpt $(du -h "$FP32_CKPT" | cut -f1), tex=${TEX} dec=${DEC}"

# ---- 1) GPU 共享锁：短持锁轮询（拿锁 -> 查 free -> 不足立即放锁）----
mkdir -p "$ROOT/.locks"
NEED_FREE=${FREE_THRESHOLD_MB:-10500}
WAIT_WINDOW=${WAIT_WINDOW_S:-14400}
deadline=$(( $(date +%s) + WAIT_WINDOW ))
say "poll for GPU (need free>=${NEED_FREE}MiB, window=${WAIT_WINDOW}s) ..."
exec 9>"$ROOT/.locks/gpu4090.lock"
while :; do
  if ! flock -n 9; then
    flock -u 9 2>/dev/null || true
    if [ "$(date +%s)" -ge "$deadline" ]; then
      say "window expired waiting for lock"
      exec 9>&-
      printf '{"status":"blocked","reason":"lock held by others","need_free_mib":%s,"window_s":%s}\n' \
        "$NEED_FREE" "$WAIT_WINDOW" > "$OUT/blocked_fp32.json"
      exit 2
    fi
    sleep $((35 + RANDOM % 30))
    continue
  fi
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -ge "$NEED_FREE" ]; then
    say "GPU lock acquired, free=${free}MiB >= ${NEED_FREE}MiB, start"
    break
  fi
  say "lock held but free=${free}MiB < ${NEED_FREE}MiB, release and retry"
  flock -u 9 2>/dev/null || true
  if [ "$(date +%s)" -ge "$deadline" ]; then
    say "window expired (free=${free}MiB)"
    exec 9>&-
    printf '{"status":"blocked","reason":"co-tenant VRAM","need_free_mib":%s,"observed_free_mib":%s,"window_s":%s}\n' \
      "$NEED_FREE" "$free" "$WAIT_WINDOW" > "$OUT/blocked_fp32.json"
    exit 2
  fi
  sleep $((35 + RANDOM % 30))
done

BASE_VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
say "baseline GPU mem used (all containers) = ${BASE_VRAM} MiB"

# 峰值显存采样（记录全卡已用，峰值减 baseline 为本进程增量）
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" >> "$OUT/logs/vram_fp32_all.txt"; sleep 2; done ) &
MON=$!
trap 'kill $MON 2>/dev/null; flock -u 9; rm -f "$OUT/blocked_fp32.json"' EXIT
rm -f "$OUT/blocked_fp32.json"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- 2) 规范输入包（CPU）----
if [ ! -f "$OUT/inputs/proxy_v0/meta.json" ] || [ ! -f "$OUT/inputs/vggt_v0/meta.json" ]; then
  say "build inputs (ref view 0)"
  "$TPY" "$ROOT/scripts/experiments/tex_t1_build_inputs.py" --ref-index 0 --out "$OUT/inputs" >> "$LOG" 2>&1 \
    || { say "build_inputs FAILED"; exit 1; }
else
  say "inputs exist, skip rebuild"
fi

# ---- 3) SUPRA-Tex fp32 推理：两个 mesh × 两个种子 ----
cd "$SRC" || exit 1
for mesh in proxy vggt; do
  for seed in 42 1234; do
    name="supra_${mesh}_v0_s${seed}"
    glb="$OUT/glb_fp32/${name}.glb"
    status="$OUT/logs/fp32_${name}.status"
    if [ -f "$glb" ]; then say "skip existing $name"; continue; fi
    if [ -f "$status" ] && grep -q "^OK" "$status" 2>/dev/null; then say "skip OK-status $name"; continue; fi
    say "run $name ..."
    t0=$(date +%s)
    "$SPY" texturing.py \
      --pretrained "$ROOT/models/supra_tex/completion" \
      --trellis-root "$ROOT/models/trellis2" \
      --input-dir "$OUT/inputs/${mesh}_v0" \
      --output "$glb" \
      --attention sdpa --low-vram \
      --seed "$seed" --texture-size "$TEX" --decimation-target "$DEC" --texture-format png \
      >> "$OUT/logs/${name}_fp32.log" 2>&1
    rc=$?
    t1=$(date +%s)
    peak=$(awk -v a="$t0" -v b="$t1" '$1>=a && $1<=b { if ($2+0>m) m=$2+0 } END { print m+0 }' "$OUT/logs/vram_fp32_all.txt")
    if [ $rc -eq 0 ] && [ -f "$glb" ]; then
      echo "OK elapsed=$((t1-t0))s peak_all=${peak}MiB peak_delta=$((peak-BASE_VRAM))MiB" > "$status"
      say "OK $name  elapsed=$((t1-t0))s peak_all=${peak}MiB peak_delta=$((peak-BASE_VRAM))MiB"
    else
      reason=$(grep -o "CUDA out of memory" "$OUT/logs/${name}_fp32.log" | head -1)
      echo "FAIL rc=${rc} elapsed=$((t1-t0))s peak_all=${peak}MiB reason=${reason:-unknown}" > "$status"
      say "FAIL $name rc=$rc elapsed=$((t1-t0))s peak_all=${peak}MiB reason=${reason:-unknown}"
    fi
  done
done

# ---- 4) T0 单视角（同一参考视图 0）对照 ----
if [ -f "$OUT/t0_single_view/summary_t1single.json" ]; then
  say "T0 single-view already done, skip"
else
  say "T0 single-view baseline (max-cams 1)"
  "$TPY" "$ROOT/scripts/experiments/texture_fixed_mesh.py" \
    --meshes proxy,vggt --atlas 2048 --max-cams 1 --variants weighted_mean,best_view \
    --tag _t1single --out "$OUT/t0_single_view" >> "$OUT/logs/t0_single_view.log" 2>&1
  say "T0 single-view rc=$?"
fi

# ---- 5) 四轴评测（与同规格评测同一 tex_t1_eval.py 口径）----
if [ -d "$OUT/t0_single_view/cache" ]; then
  specs="--glb t0:proxy --glb t0:vggt"
  for mesh in proxy vggt; do
    for seed in 42 1234; do
      f="$OUT/glb_fp32/supra_${mesh}_v0_s${seed}.glb"
      [ -f "$f" ] && specs="$specs --glb supra_${mesh}_v0_s${seed}=$f"
    done
  done
  say "eval four-axis (fp32)"
  "$TPY" "$ROOT/scripts/experiments/tex_t1_eval.py" $specs --out "$OUT/eval_fp32" >> "$OUT/logs/eval_fp32.log" 2>&1
  say "eval rc=$?"
fi

say "DONE"
