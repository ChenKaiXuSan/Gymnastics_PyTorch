#!/bin/bash
# Strict external baselines that are trained per fold (fusion.external.published).
#
#   qsub -v METHOD=canonpose,DATASET=freeman,FOLDS="fold_01" pegasus/external_trained_qsub.sh
#   qsub -v METHOD=canonpose,DATASET=gymnastics,STAGE=all pegasus/external_trained_qsub.sh
#
# METHOD   canonpose | metapose        (required)
# DATASET  gymnastics | freeman | sportspose   (required)
# STAGE    all (default) | prepare | train | evaluate   (canonpose)
#          all | prepare | s1 | train | evaluate          (metapose)
#          several stages run in order when "::"-separated, e.g. STAGE=prepare::s1
# FOLDS    space-separated fold names to train (default: all)
# EXTRA    extra command-line arguments, "::"-separated (qsub -v cannot carry spaces),
#          e.g. EXTRA="--epochs::3::--force"
# REPO     checkout to run (default: the main working tree)
#
#PBS -A HP260146
#PBS -q gen_S
#PBS -l gpunum_job=1
#PBS -l elapstim_req=06:00:00
#PBS -b 1
#PBS -N ext_train
#PBS -j o

set -u
REPO="${REPO:-/work/HP260146/chenkaixu/Gymnastics_PyTorch}"
cd "$REPO" || exit 1
: "${METHOD:?METHOD must be passed via qsub -v}"
: "${DATASET:?DATASET must be passed via qsub -v}"
STAGE="${STAGE:-all}"
export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GYMNASTICS_DATA_ROOT=/work/HP260146/chenkaixu/gymnastics
ENVBIN=/home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body/bin
export PATH="$ENVBIN:$PATH"
ARGS=(--dataset "$DATASET")
[ -n "${FOLDS:-}" ] && ARGS+=(--folds ${FOLDS//::/ })
if [ -n "${EXTRA:-}" ]; then
  IFS=$'\x1f' read -r -a EXTRA_ARGS <<< "${EXTRA//::/$'\x1f'}"
  ARGS+=("${EXTRA_ARGS[@]}")
fi
echo "[ext_train] host=$(hostname) method=$METHOD dataset=$DATASET stage=$STAGE folds=${FOLDS:-all} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[ext_train] nvidia-smi unavailable"
status=0
for stage in ${STAGE//::/ }; do
  echo "[ext_train] stage=$stage start=$(date -Is)"
  "$ENVBIN/python" -m fusion external-published "$METHOD" --stage "$stage" "${ARGS[@]}"
  status=$?
  [ $status -ne 0 ] && break
done
echo "[ext_train] exit=$status end=$(date -Is)"
exit $status
