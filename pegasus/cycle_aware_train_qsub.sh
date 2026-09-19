#!/bin/bash
# Cycle-aware fusion: one training run with the dataset's default split
# (private data: the fixed 96/27/14 paper split) on the gpu queue.
#
#   qsub -o local/runs/cycle_aware/joblogs/gym_v1_seed0.log \
#        -v DATA=gymnastics,RUN_NAME=gym_v1_seed0 pegasus/cycle_aware_train_qsub.sh
#
# Variables (qsub -v):
#   DATA       gymnastics | freeman | synthetic  (required)
#   RUN_NAME   run dir = local/runs/cycle_aware/<RUN_NAME>  (default <DATA>_v1_seed<SEED>)
#   EXPERIMENT optional Hydra experiment preset (no_film, no_cross_view, ...)
#   SEED       seed (default 0)
#   EPOCHS     epochs (default 50)
#   OVERRIDES  extra Hydra overrides separated by "|" (qsub -v values cannot
#              contain spaces), e.g. "model.hidden_dim=256|loss.periodicity_weight=0.3"
#
#PBS -A HP260146
#PBS -q gen_S
#PBS -l gpunum_job=1
#PBS -l elapstim_req=12:00:00
#PBS -b 1
#PBS -N ca_train
#PBS -j o

set -u
REPO=/work/HP260146/chenkaixu/Gymnastics_PyTorch
cd "$REPO" || exit 1
: "${DATA:?DATA must be passed via qsub -v}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"
RUN_NAME="${RUN_NAME:-${DATA}_v1_seed${SEED}}"
EXPERIMENT="${EXPERIMENT:-}"
OVERRIDES="${OVERRIDES:-}"

export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GYMNASTICS_DATA_ROOT=/work/HP260146/chenkaixu/gymnastics

ENVBIN=/home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body/bin
export PATH="$ENVBIN:$PATH"
PYBIN="$ENVBIN/python"

echo "[ca_train] host=$(hostname) data=$DATA run=$RUN_NAME seed=$SEED epochs=$EPOCHS experiment=${EXPERIMENT:-none} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[ca_train] nvidia-smi unavailable"

ARGS=("data=$DATA" "run_name=$RUN_NAME" "seed=$SEED" "trainer.max_epochs=$EPOCHS"
      "trainer.enable_progress_bar=false" "data.num_workers=${NUM_WORKERS:-0}")
[ -n "$EXPERIMENT" ] && ARGS+=("experiment=$EXPERIMENT")
if [ -n "$OVERRIDES" ]; then
  # "|"-separated list -> one argument per override (spaces are also accepted).
  IFS='| ' read -r -a EXTRA_ARGS <<<"$OVERRIDES"
  ARGS+=("${EXTRA_ARGS[@]}")
fi

"$PYBIN" -u -m gymnastics fuse cycle-aware "${ARGS[@]}"
status=$?
echo "[ca_train] run=$RUN_NAME exit=$status end=$(date -Is)"
exit $status
