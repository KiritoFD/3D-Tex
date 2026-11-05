#!/bin/bash
# Tex-T1: GPU 排队 + 固定 Mesh SUPRA-Tex 推理 + T0 单视角对照
set -uo pipefail
ROOT=/root/3dtex
OUT=$ROOT/outputs/tex_t1_supra_20261105
SPY=/opt/conda/envs/supra-tex/bin/python
TPY=/opt/conda/envs/3dtex/bin/python
SRC=$ROOT/third_party/supra_tex
LOG=$OUT/logs/run.log
mkdir -p "$OUT/logs" "$OUT/glb" "$OUT/inputs"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---- GPU 共享锁：轮询"拿锁 -> 查 free -> 不足立即放锁"，等待期间不占锁 ----
mkdir -p "$ROOT/.locks"
NEED_FREE=${NEED_FREE_MB:-10500}
WAIT_WINDOW=${WAIT_WINDOW_S:-5400}
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
        "$NEED_FREE" "$WAIT_WINDOW" > "$OUT/blocked.json"
      exit 2
    fi
    sleep 45
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
      "$NEED_FREE" "$free" "$WAIT_WINDOW" > "$OUT/blocked.json"
    exit 2
  fi
  sleep 45
done

BASE_VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
say "baseline GPU mem used (all containers) = ${BASE_VRAM} MiB"

# 峰值显存采样（记录的是全卡已用，需减去 baseline）
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" >> "$OUT/logs/vram_samples.txt"; sleep 2; done ) &
MON=$!
trap 'kill $MON 2>/dev/null; flock -u 9; rm -f "$OUT/blocked.json"' EXIT
rm -f "$OUT/blocked.json"

# ---- 1) 规范输入包 ----
say "build inputs (ref view 0)"
"$TPY" "$ROOT/scripts/experiments/tex_t1_build_inputs.py" --ref-index 0 --out "$OUT/inputs" >> "$LOG" 2>&1 \
  || { say "build_inputs FAILED"; exit 1; }

# ---- 2) SUPRA-Tex 推理：两个 mesh × 两个种子 ----
cd "$SRC" || exit 1
for mesh in proxy vggt; do
  for seed in 42 1234; do
    name="supra_${mesh}_v0_s${seed}"
    glb="$OUT/glb/${name}.glb"
    if [ -f "$glb" ]; then say "skip existing $name"; continue; fi
    say "run $name ..."
    t0=$(date +%s)
    "$SPY" texturing.py \
      --pretrained "$ROOT/models/supra_tex/completion" \
      --trellis-root "$ROOT/models/trellis2" \
      --input-dir "$OUT/inputs/${mesh}_v0" \
      --output "$glb" \
      --attention sdpa --low-vram \
      --seed "$seed" --texture-size 2048 --texture-format png \
      >> "$OUT/logs/${name}.log" 2>&1
    rc=$?
    t1=$(date +%s)
    peak=$(awk -v a="$t0" -v b="$t1" '$1>=a && $1<=b { if ($2+0>m) m=$2+0 } END { print m+0 }' "$OUT/logs/vram_samples.txt")
    if [ $rc -eq 0 ] && [ -f "$glb" ]; then
      say "OK $name  elapsed=$((t1-t0))s peak_all=${peak}MiB peak_delta=$((peak-BASE_VRAM))MiB"
    else
      say "FAIL $name rc=$rc elapsed=$((t1-t0))s peak_all=${peak}MiB"
    fi
  done
done

# ---- 3) T0 单视角（同一参考视图 0）对照 ----
if [ -f "$OUT/t0_single_view/summary_t1single.json" ]; then
  say "T0 single-view already done, skip"
else
  say "T0 single-view baseline (max-cams 1)"
  "$TPY" "$ROOT/scripts/experiments/texture_fixed_mesh.py" \
    --meshes proxy,vggt --atlas 2048 --max-cams 1 --variants weighted_mean,best_view \
    --tag _t1single --out "$OUT/t0_single_view" >> "$OUT/logs/t0_single_view.log" 2>&1
  say "T0 single-view rc=$?"
fi

# ---- 4) 四轴评测（T0 单视角同一口径 + 已产出的 SUPRA GLB） ----
if [ -d "$OUT/t0_single_view/cache" ]; then
  specs="--glb t0:proxy --glb t0:vggt"
  for mesh in proxy vggt; do
    for seed in 42 1234; do
      f="$OUT/glb/supra_${mesh}_v0_s${seed}.glb"
      [ -f "$f" ] && specs="$specs --glb supra_${mesh}_v0_s${seed}=$f"
    done
  done
  say "eval four-axis"
  "$SPY" "$ROOT/scripts/experiments/tex_t1_eval.py" $specs --out "$OUT/eval" >> "$OUT/logs/eval.log" 2>&1
  say "eval rc=$?"
fi

say "DONE"