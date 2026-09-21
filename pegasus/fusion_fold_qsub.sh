#!/bin/bash
# Cycle-aware fusion: train + evaluate ONE cross-validation fold on the gpu queue.
#
#   qsub -o local/runs/cycle_aware/joblogs/gym_v1_5fold_fold_01.log \
#        -v DATA=gymnastics,FOLD=fold_01,SWEEP=gym_v1_5fold pegasus/fusion_fold_qsub.sh
#
# Variables (qsub -v):
#   DATA       gymnastics | freeman | sportspose (required)
#   FOLD       fold_01 .. fold_05              (required)
#   SWEEP      sweep name; run dir = local/runs/cycle_aware/<SWEEP>/<FOLD>  (default <DATA>_v1_5fold)
#   FOLDS_DIR  fold-file directory (default: src/configs/fusion/folds/gymnastics
#              or src/configs/shared/folds/freeman; src/configs/fusion/folds/freeman_all40
#              for the 40-subject FreeMan protocol)
#   EXPERIMENT optional Hydra experiment preset (no_film, no_cross_view, ...)
#   SEED       seed (default 0)
#   EPOCHS     epochs (default 50)
#   NUM_WORKERS DataLoader workers (default 8; the project decision of 2026-09-21 is
#              to always train with 8 workers, never 0)
#   OVERRIDES  extra Hydra overrides separated by "::" (qsub -v values cannot
#              contain spaces or shell characters), e.g.
#              "test_only=true::checkpoint=path/last.ckpt"
#
# Usually submitted for all folds at once by pegasus/submit_fusion_5fold.sh.
#
#PBS -A HP260146
#PBS -q gen_S
#PBS -l gpunum_job=1
# One fold takes ~1 h; a short request backfills into scheduling gaps.
#PBS -l elapstim_req=03:00:00
#PBS -b 1
#PBS -N ca_fold
#PBS -j o

set -u
REPO="${REPO:-/work/HP260146/chenkaixu/Gymnastics_PyTorch}"
cd "$REPO" || exit 1
: "${DATA:?DATA (gymnastics|freeman|sportspose) must be passed via qsub -v}"
: "${FOLD:?FOLD (fold_01..fold_05) must be passed via qsub -v}"
SWEEP="${SWEEP:-${DATA}_v1_5fold}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"
EXPERIMENT="${EXPERIMENT:-}"
OVERRIDES="${OVERRIDES:-}"
case "$DATA" in
  gymnastics) DEFAULT_FOLDS=src/configs/fusion/folds/gymnastics ;;
  freeman)    DEFAULT_FOLDS=src/configs/shared/folds/freeman ;;
  sportspose) DEFAULT_FOLDS=src/configs/fusion/folds/sportspose ;;
  *) echo "[ca_fold] unsupported DATA=$DATA"; exit 2 ;;
esac
FOLDS_DIR="${FOLDS_DIR:-$DEFAULT_FOLDS}"
FOLD_JSON="$FOLDS_DIR/$FOLD.json"
[ -f "$FOLD_JSON" ] || { echo "[ca_fold] missing fold file $FOLD_JSON"; exit 2; }

export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GYMNASTICS_DATA_ROOT=/work/HP260146/chenkaixu/gymnastics

ENVBIN=/home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body/bin
export PATH="$ENVBIN:$PATH"
PYBIN="$ENVBIN/python"

echo "[ca_fold] host=$(hostname) data=$DATA fold=$FOLD sweep=$SWEEP seed=$SEED epochs=$EPOCHS experiment=${EXPERIMENT:-none} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[ca_fold] nvidia-smi unavailable"

ARGS=("data=$DATA" "data.fold_json=$FOLD_JSON" "run_name=$SWEEP/$FOLD" "seed=$SEED"
      "trainer.max_epochs=$EPOCHS" "trainer.enable_progress_bar=false" "data.num_workers=${NUM_WORKERS:-8}")
[ -n "$EXPERIMENT" ] && ARGS+=("experiment=$EXPERIMENT")
if [ -n "$OVERRIDES" ]; then
  # "::"-separated list -> one argument per override (spaces are also accepted).
  IFS=' ' read -r -a EXTRA_ARGS <<<"${OVERRIDES//::/ }"
  ARGS+=("${EXTRA_ARGS[@]}")
fi

"$PYBIN" -u -m fusion train "${ARGS[@]}"
status=$?
echo "[ca_fold] data=$DATA fold=$FOLD exit=$status end=$(date -Is)"
exit $status
