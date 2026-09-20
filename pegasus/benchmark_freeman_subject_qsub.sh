#!/bin/bash
# FreeMan benchmark: process one subject on the cluster gpu queue.
#
#   qsub -v SUBJECT=1,FRAME_STRIDE=25 pegasus/benchmark_freeman_subject_qsub.sh
#
# SUBJECT (required): FreeMan subject ID 1..40.
# FRAME_STRIDE (optional): overrides the config stride (default: config value).
# CONFIG (optional): benchmark YAML path (default src/configs/benchmarks/freeman_cluster.yaml).
# The pipeline caches per-session results, so resubmitting the same subject
# resumes instead of recomputing.
#
#PBS -A SKIING
#PBS -q gpu
#PBS -l gpunum_job=1
#PBS -l elapstim_req=24:00:00
#PBS -b 1
#PBS -N freeman_subj
#PBS -j o

set -u
REPO=/work/HP260146/chenkaixu/Gymnastics_PyTorch
cd "$REPO" || exit 1

: "${SUBJECT:?SUBJECT (1..40) must be passed via qsub -v}"
CONFIG="${CONFIG:-src/configs/benchmarks/freeman_cluster.yaml}"
STRIDE_ARGS=()
if [ -n "${FRAME_STRIDE:-}" ]; then
  STRIDE_ARGS=(--frame-stride "$FRAME_STRIDE")
fi
# FORCE=<stage>: pass --force-stage (e.g. infer) to redo a subject whose
# run_state entry says complete but whose caches have a different identity.
if [ -n "${FORCE:-}" ]; then
  STRIDE_ARGS+=(--force-stage "$FORCE")
fi

export PYTHONPATH="$REPO/src"
# All model weights are local or in shared caches; avoid hub traffic on
# compute nodes that may have no internet access.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[freeman_qsub] host=$(hostname) subject=$SUBJECT stride=${FRAME_STRIDE:-config} config=$CONFIG start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[freeman_qsub] nvidia-smi unavailable"
# GPULOG=<path>: sample GPU utilization every 30 s into that file.
if [ -n "${GPULOG:-}" ]; then
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 30 > "$GPULOG" 2>/dev/null &
  GPULOG_PID=$!
  trap '[ -n "${GPULOG_PID:-}" ] && kill "$GPULOG_PID" 2>/dev/null' EXIT
fi

# Compute nodes carry a system conda whose env dirs differ from the login
# node; call the env's interpreter directly instead of resolving via conda.
# The env's bin must also be on PATH so the pipeline can shell out to 7z.
ENVBIN=/home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body/bin
export PATH="$ENVBIN:$PATH"
PYBIN="$ENVBIN/python"
"$PYBIN" -m fusion benchmark-freeman run \
  --config "$CONFIG" \
  --subject "$SUBJECT" \
  "${STRIDE_ARGS[@]}"
status=$?

echo "[freeman_qsub] subject=$SUBJECT exit=$status end=$(date -Is)"
exit $status
