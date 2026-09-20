#!/bin/bash
# Submit the 5 cross-validation folds of one dataset as 5 gpu jobs.
#
#   bash pegasus/submit_cycle_aware_5fold.sh gymnastics                 # sweep gymnastics_v1_5fold
#   bash pegasus/submit_cycle_aware_5fold.sh freeman                    # sweep freeman_v1_5fold (17-subject protocol)
#   bash pegasus/submit_cycle_aware_5fold.sh freeman_all40              # all 40 subjects, session-balanced folds
#   bash pegasus/submit_cycle_aware_5fold.sh gymnastics no_film         # ablation preset -> gymnastics_v1_no_film_5fold
#   SEED=1 bash pegasus/submit_cycle_aware_5fold.sh gymnastics          # other seed
#   OVERRIDES="model.hidden_dim=256::loss.periodicity_weight=0" bash pegasus/submit_cycle_aware_5fold.sh gymnastics
#   ACCOUNT=HP260146 bash pegasus/submit_cycle_aware_5fold.sh gymnastics no_film   # HP260146 project, gen_S queue
#
# After the jobs finish, summarise with
#   PYTHONPATH=src python -m fusion.summarize local/runs/cycle_aware/<sweep>
set -eu
DATA="${1:?usage: submit_cycle_aware_5fold.sh <gymnastics|freeman|freeman_all40> [experiment]}"
EXPERIMENT="${2:-}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"
SWEEP="${SWEEP:-${1}_v1${EXPERIMENT:+_$EXPERIMENT}_5fold${SEED:+_seed$SEED}}"
PROTOCOL="$DATA"
case "$DATA" in
  gymnastics)    FOLDS_DIR="${FOLDS_DIR:-src/configs/fusion/folds/gymnastics}" ;;
  freeman)       FOLDS_DIR="${FOLDS_DIR:-src/configs/shared/folds/freeman}" ;;
  freeman_all40) FOLDS_DIR="${FOLDS_DIR:-src/configs/fusion/folds/freeman_all40}"; DATA=freeman ;;
  *) echo "unsupported dataset $DATA"; exit 2 ;;
esac
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
# ACCOUNT=SKIING (default) -> gpu queue; ACCOUNT=HP260146 -> gen_S queue (each account has its own request cap).
JOB_SCRIPT=pegasus/cycle_aware_fold_qsub.sh
[ "${ACCOUNT:-SKIING}" = "HP260146" ] && JOB_SCRIPT=pegasus/cycle_aware_fold_qsub_hp.sh
mkdir -p local/runs/cycle_aware/joblogs
for fold_file in "$FOLDS_DIR"/fold_*.json; do
  FOLD="$(basename "$fold_file" .json)"
  LOG="local/runs/cycle_aware/joblogs/${SWEEP}_${FOLD}.log"
  VARS="DATA=$DATA,FOLD=$FOLD,SWEEP=$SWEEP,FOLDS_DIR=$FOLDS_DIR,SEED=$SEED,EPOCHS=$EPOCHS"
  [ -n "$EXPERIMENT" ] && VARS="$VARS,EXPERIMENT=$EXPERIMENT"
  [ -n "${OVERRIDES:-}" ] && VARS="$VARS,OVERRIDES=$OVERRIDES"
  echo "qsub -N ca_${FOLD} -o $LOG -v $VARS $JOB_SCRIPT"
  qsub -N "ca_${FOLD}" -o "$LOG" -v "$VARS" "$JOB_SCRIPT"
done
echo "submitted 5 folds of $DATA -> local/runs/cycle_aware/$SWEEP/<fold>; summarise with:"
echo "  PYTHONPATH=src python -m fusion.summarize local/runs/cycle_aware/$SWEEP"
