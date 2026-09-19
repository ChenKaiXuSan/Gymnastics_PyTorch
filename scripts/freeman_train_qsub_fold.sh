#!/bin/bash
# Subject-disjoint FreeMan training: train one fold on the cluster gpu queue.
#
#   qsub -v FOLD=fold_01,SEED=0 scripts/freeman_train_qsub_fold.sh
#
# FOLD (required): fold name under configs/fusion/folds/freeman (fold_01..fold_05).
# SEED (optional): training seed (default 0); becomes part of the run id.
# CONFIG (optional): rotation-aware YAML (default configs/fusion/rotation_aware_freeman.yaml).
# Prerequisite (login node, CPU):
#   gymnastics benchmark freeman-train prepare-cache --config $CONFIG
#   gymnastics benchmark freeman-train write-folds   --config $CONFIG
#
#PBS -A SKIING
#PBS -q gpu
#PBS -l gpunum_job=1
#PBS -l elapstim_req=24:00:00
#PBS -b 1
#PBS -N freeman_train
#PBS -j o

set -u
REPO=/work/HP260146/chenkaixu/Gymnastics_PyTorch
cd "$REPO" || exit 1

: "${FOLD:?FOLD (fold_01..fold_05) must be passed via qsub -v}"
SEED="${SEED:-0}"
CONFIG="${CONFIG:-configs/fusion/rotation_aware_freeman.yaml}"
RUN_ID="freeman_${FOLD}_a6_e12_s${SEED}"

export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[freeman_train] host=$(hostname) fold=$FOLD seed=$SEED run_id=$RUN_ID start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[freeman_train] nvidia-smi unavailable"

ENVBIN=/home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body/bin
export PATH="$ENVBIN:$PATH"
PYBIN="$ENVBIN/python"

# The trainer reads training.seed from the config; override through a
# seed-specific resolved copy so the run id and the seed always agree.
RESOLVED="local/runs/fuse_rotation_aware_freeman/configs/${RUN_ID}.yaml"
mkdir -p "$(dirname "$RESOLVED")"
"$PYBIN" - "$CONFIG" "$RESOLVED" "$SEED" <<'EOF'
import sys, yaml
source, target, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
config = yaml.safe_load(open(source, encoding="utf-8"))
config["training"]["seed"] = seed
yaml.safe_dump(config, open(target, "w", encoding="utf-8"), sort_keys=False)
EOF

"$PYBIN" -m gymnastics fuse rotation-aware train \
  --config "$RESOLVED" \
  --fold "configs/fusion/folds/freeman/${FOLD}.json" \
  --run-id "$RUN_ID" \
  --ablation A6
status=$?

echo "[freeman_train] fold=$FOLD run_id=$RUN_ID exit=$status end=$(date -Is)"
exit $status
