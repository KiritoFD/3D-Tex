#!/usr/bin/env bash
# Tex-T2 GPU 排队 + 分档回退执行器。
#
#   bash scripts/experiments/tex_t2_run.sh [mode] [meshes]
#     mode   full  : 512, 标准管线；OOM 后依次回退 384/256 + CPU offload
#            lowmem: 直接从 384 + CPU offload 开始
#     meshes 默认 proxy,vggt
#
# 排队：所有 GPU 工作都在 /root/3dtex/.locks/gpu4090.lock 的 flock 下互斥进行；
# 拿到锁后若显存仍不足，释放锁并等待，绝不杀他人进程。
set -uo pipefail

ROOT=/root/3dtex
OUT=$ROOT/outputs/tex_t2_hunyuan_20261105
LOCK=$ROOT/.locks/gpu4090.lock
PREFIX=/root/conda_prefix/hy3d_paint
PY=$PREFIX/bin/python
LOG=$OUT/logs
MODE=${1:-full}
MESHES=${2:-proxy,vggt}
DEADLINE=$(( $(date +%s) + 3600 ))

export CUDA_HOME=$PREFIX
export PATH=$PREFIX/bin:$PATH
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONPATH=$ROOT/third_party/stubs:$ROOT/third_party/hunyuan3d/hy3dpaint
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOG"

need_mib() {
  case "$1" in
    512) echo 19500 ;;
    384) echo 5500 ;;   # 384 + cpu-offload + dino-cpu 估算
    256) echo 4000 ;;   # 256 + cpu-offload + dino-cpu 估算
    *) echo 19500 ;;
  esac
}
res_wait_s() {
  case "$1" in 512) echo 900 ;; *) echo 300 ;; esac
}

free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '
}

run_attempt() {
  local mesh=$1 res=$2 extra=$3
  local tag="_r${res}"
  echo "[run] mesh=$mesh res=$res extra='$extra' $(date '+%F %T')" | tee -a "$LOG/run_attempts.log"
  $PY "$ROOT/scripts/experiments/tex_t2_hunyuan.py" \
      --stage all --meshes "$mesh" --resolution "$res" \
      --render-size 1024 --texture-size 2048 --tag "$tag" $extra \
      2>&1 | tee -a "$LOG/paint_eval_${mesh}${tag}.log"
  return "${PIPESTATUS[0]}"
}

attempt_mesh() {
  local mesh=$1
  local plans
  if [ "$MODE" = "lowmem" ]; then
    plans="384 256"
  else
    plans="512 384 256"
  fi
  for res in $plans; do
    local want=$(need_mib "$res")
    local extra=""
    [ "$res" -lt 512 ] && extra="--cpu-offload --dino-cpu"
    local res_deadline=$(( $(date +%s) + $(res_wait_s "$res") ))

    while true; do
      if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        echo "[wait] ${mesh} res=$res timeout waiting for free VRAM" | tee -a "$LOG/run_attempts.log"
        return 3
      fi
      if [ "$(date +%s)" -gt "$res_deadline" ]; then
        echo "[wait] ${mesh} res=$res no enough VRAM in window, fall back" | tee -a "$LOG/run_attempts.log"
        break
      fi
      exec 9>"$LOCK"
      if flock -w 3600 9; then
        free=$(free_mib)
        echo "[lock] acquired, free=${free}MiB need=${want}MiB" | tee -a "$LOG/run_attempts.log"
        if [ "${free:-0}" -ge "$want" ]; then
          run_attempt "$mesh" "$res" "$extra"
          rc=$?
          flock -u 9
          exec 9>&-
          if [ "$rc" -eq 0 ]; then
            return 0
          fi
          echo "[retry] ${mesh} res=$res rc=$rc" | tee -a "$LOG/run_attempts.log"
          break
        fi
        flock -u 9
        exec 9>&-
      else
        echo "[lock] timeout waiting for gpu lock" | tee -a "$LOG/run_attempts.log"
      fi
      sleep 60
    done
  done
  return 1
}

overall=0
for m in ${MESHES//,/ }; do
  attempt_mesh "$m"
  rc=$?
  [ "$rc" -ne 0 ] && overall=$rc
done
echo "[done] overall=$overall $(date '+%F %T')" | tee -a "$LOG/run_attempts.log"
exit "$overall"
