#!/bin/bash
# Strict external baselines (fusion.external.published) on one GPU node.
#
#   qsub -v STAGE=keypoints2d,DATASET=gymnastics pegasus/external_published_qsub.sh   # private 2D cache (CPU-bound)
#   qsub -v STAGE=videopose3d,DATASET=freeman    pegasus/external_published_qsub.sh   # lift + evaluate (5 folds)
#   qsub -v STAGE=videopose3d,DATASET=gymnastics,MODE=per_view pegasus/external_published_qsub.sh
#
# STAGE    keypoints2d | videopose3d            (required)
# DATASET  gymnastics | freeman | sportspose    (required)
# MODE     procrustes_average (default) | per_view      (videopose3d only)
# EXTRA    extra arguments appended to the command
# REPO     checkout to run (default: the main working tree)
#
#PBS -A SKIING
#PBS -q gpu
#PBS -l gpunum_job=1
#PBS -l elapstim_req=03:00:00
#PBS -b 1
#PBS -N ext_pub
#PBS -j o

set -u
REPO="${REPO:-/work/HP260146/chenkaixu/Gymnastics_PyTorch}"
cd "$REPO" || exit 1
: "${STAGE:?STAGE (keypoints2d|videopose3d) must be passed via qsub -v}"
: "${DATASET:?DATASET (gymnastics|freeman|sportspose) must be passed via qsub -v}"
MODE="${MODE:-procrustes_average}"

export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GYMNASTICS_DATA_ROOT=/work/HP260146/chenkaixu/gymnastics

ENVBIN=/home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body/bin
export PATH="$ENVBIN:$PATH"
PYBIN="$ENVBIN/python"

echo "[ext_pub] host=$(hostname) stage=$STAGE dataset=$DATASET mode=$MODE start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[ext_pub] nvidia-smi unavailable"
case "$STAGE" in
  keypoints2d) "$PYBIN" -m fusion external-published keypoints2d --dataset "$DATASET" --workers 8 ${EXTRA:-} ;;
  videopose3d) "$PYBIN" -m fusion external-published videopose3d --dataset "$DATASET" --mode "$MODE" --device cuda ${EXTRA:-} ;;
  *) echo "[ext_pub] unknown STAGE=$STAGE"; exit 2 ;;
esac
status=$?
echo "[ext_pub] exit=$status end=$(date -Is)"
exit $status
