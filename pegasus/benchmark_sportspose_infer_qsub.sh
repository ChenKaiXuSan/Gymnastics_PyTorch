#!/bin/bash
# SportsPose benchmark: SAM3D-Body on the two selected views of a few subjects.
#
#   qsub -v DAYS=indoors,SUBJECTS="S00 S01 S02 S03" pegasus/benchmark_sportspose_infer_qsub.sh
#
# DAYS (optional):     indoors | outdoors (default: both)
# SUBJECTS (optional): space-separated S-ids (default: every subject of DAYS)
# CONFIG (optional):   benchmark YAML (default src/configs/benchmarks/sportspose.yaml)
# REPO (optional):     checkout to run (default: the main working tree)
# Run `python -m fusion benchmark-sportspose select-views` once before submitting.
# The cache is per clip and view, so resubmitting resumes.
#
#PBS -A HP260146
#PBS -q gen_S
#PBS -l gpunum_job=1
#PBS -l elapstim_req=24:00:00
#PBS -b 1
#PBS -N sportspose_infer
#PBS -j o

set -u
REPO="${REPO:-/work/HP260146/chenkaixu/Gymnastics_PyTorch}"
cd "$REPO" || exit 1
CONFIG="${CONFIG:-src/configs/benchmarks/sportspose.yaml}"
ARGS=()
[ -n "${DAYS:-}" ] && ARGS+=(--days $DAYS)
[ -n "${SUBJECTS:-}" ] && ARGS+=(--subjects $SUBJECTS)

export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GYMNASTICS_DATA_ROOT=/work/HP260146/chenkaixu/gymnastics

ENVBIN=/home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body/bin
export PATH="$ENVBIN:$PATH"
PYBIN="$ENVBIN/python"

echo "[sportspose_qsub] host=$(hostname) days=${DAYS:-all} subjects=${SUBJECTS:-all} config=$CONFIG start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[sportspose_qsub] nvidia-smi unavailable"
"$PYBIN" -m fusion benchmark-sportspose --config "$CONFIG" infer --device 0 "${ARGS[@]}"
status=$?
echo "[sportspose_qsub] exit=$status end=$(date -Is)"
exit $status
