#!/bin/bash
# Out-of-fold evaluation + comparison of the FreeMan-trained folds on a
# compute node (the login node's 16 GB user quota is too small for the
# session-level inference of 552 sessions).
#
#   qsub -o local/runs/freeman_trained_fusion/joblogs/evaluate_s0.log -v SEED=0 scripts/freeman_train_eval_qsub.sh
#
#PBS -A SKIING
#PBS -q gpu
#PBS -l gpunum_job=1
#PBS -l elapstim_req=12:00:00
#PBS -b 1
#PBS -N freeman_eval
#PBS -j o

set -u
REPO=/work/HP260146/chenkaixu/Gymnastics_PyTorch
cd "$REPO" || exit 1
SEED="${SEED:-0}"
CONFIG="${CONFIG:-src/configs/archive/rotation_aware_freeman.yaml}"
# STAGE=evaluate (default): out-of-fold evaluation + compare of the trained folds.
# STAGE=zero-shot: score one private checkpoint (RUN_ID, ROTATION_CONFIG) zero-shot.
STAGE="${STAGE:-evaluate}"

export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
ENVBIN=/home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body/bin
export PATH="$ENVBIN:$PATH"
PYBIN="$ENVBIN/python"

echo "[freeman_eval] host=$(hostname) stage=$STAGE seed=$SEED run_id=${RUN_ID:-} start=$(date -Is)"
if [ "$STAGE" = "zero-shot" ]; then
  : "${RUN_ID:?RUN_ID must be passed for STAGE=zero-shot}"
  "$PYBIN" -m fusion benchmark-freeman-train zero-shot --config "$CONFIG" \
    --run-id "$RUN_ID" --rotation-config "${ROTATION_CONFIG:-src/configs/archive/rotation_aware.yaml}"
  status=$?
else
  "$PYBIN" -m fusion benchmark-freeman-train evaluate --config "$CONFIG" --seed "$SEED"
  status=$?
  if [ $status -eq 0 ]; then
    "$PYBIN" -m fusion benchmark-freeman-train compare --config "$CONFIG" --seed "$SEED"
    status=$?
  fi
fi
echo "[freeman_eval] seed=$SEED exit=$status end=$(date -Is)"
exit $status
