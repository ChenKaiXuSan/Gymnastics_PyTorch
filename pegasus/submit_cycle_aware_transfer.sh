#!/bin/bash
# Evaluate ONE existing checkpoint on every fold of a target protocol (zero-shot
# transfer, no training): 5 short gpu jobs writing <sweep>/fold_NN/result.json.
#
#   bash pegasus/submit_cycle_aware_transfer.sh <checkpoint> <gymnastics|freeman|freeman_all40> [sweep_name] [extra overrides]
#   e.g.
#   bash pegasus/submit_cycle_aware_transfer.sh \
#        local/runs/cycle_aware/freeman_all40_v1_reference_supervised_5fold_seed0/fold_01/checkpoints/last.ckpt \
#        gymnastics freeman_all40_refsup_f01_to_gymnastics
#
# Summarise afterwards with
#   PYTHONPATH=src python -m gymnastics.fusion.cycle_aware.summarize local/runs/cycle_aware/<sweep_name>
set -eu
CKPT="${1:?checkpoint path}"
TARGET="${2:?target protocol: gymnastics|freeman|freeman_all40}"
SWEEP="${3:-transfer_$(basename "$(dirname "$(dirname "$CKPT")")")_to_${TARGET}}"
EXTRA="${4:-}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
[ -f "$CKPT" ] || { echo "missing checkpoint $CKPT"; exit 2; }
export SWEEP
OVERRIDES="test_only=true checkpoint=$CKPT ${EXTRA}" bash pegasus/submit_cycle_aware_5fold.sh "$TARGET"
