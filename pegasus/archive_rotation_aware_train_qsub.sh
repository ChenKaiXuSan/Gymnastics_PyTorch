#!/bin/bash
# Generic rotation-aware family training job for the cluster gpu queue.
#
#   qsub -o local/runs/fuse_rotation_aware/joblogs/<run_id>.log \
#        -v CONFIG=src/configs/archive/rotation_aware_plain_tcn.yaml,RUN_ID=all137_b1_e100_seed0,ABLATION=B1 \
#        pegasus/archive_rotation_aware_train_qsub.sh
#
# CONFIG, RUN_ID, ABLATION (required); FOLD (optional, passed to --fold);
# SEED (optional, overrides training.seed through a resolved config copy).
#
#PBS -A SKIING
#PBS -q gpu
#PBS -l gpunum_job=1
#PBS -l elapstim_req=24:00:00
#PBS -b 1
#PBS -N ra_train
#PBS -j o

set -u
REPO=/work/HP260146/chenkaixu/Gymnastics_PyTorch
cd "$REPO" || exit 1
: "${CONFIG:?CONFIG must be passed via qsub -v}"
: "${RUN_ID:?RUN_ID must be passed via qsub -v}"
: "${ABLATION:?ABLATION must be passed via qsub -v}"

export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GYMNASTICS_DATA_ROOT=/work/HP260146/chenkaixu/gymnastics

echo "[ra_train] host=$(hostname) config=$CONFIG run_id=$RUN_ID ablation=$ABLATION start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[ra_train] nvidia-smi unavailable"

ENVBIN=/home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body/bin
export PATH="$ENVBIN:$PATH"
PYBIN="$ENVBIN/python"

ACTIVE_CONFIG="$CONFIG"
if [ -n "${SEED:-}" ]; then
  ACTIVE_CONFIG="local/runs/fuse_rotation_aware/configs/${RUN_ID}.yaml"
  mkdir -p "$(dirname "$ACTIVE_CONFIG")"
  "$PYBIN" - "$CONFIG" "$ACTIVE_CONFIG" "$SEED" <<'EOF'
import sys, yaml
source, target, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
config = yaml.safe_load(open(source, encoding="utf-8"))
config["training"]["seed"] = seed
yaml.safe_dump(config, open(target, "w", encoding="utf-8"), sort_keys=False)
EOF
fi

FOLD_ARGS=()
if [ -n "${FOLD:-}" ]; then
  FOLD_ARGS=(--fold "$FOLD")
fi

"$PYBIN" -m fusion rotation-aware train \
  --config "$ACTIVE_CONFIG" "${FOLD_ARGS[@]}" \
  --run-id "$RUN_ID" --ablation "$ABLATION"
status=$?
echo "[ra_train] run_id=$RUN_ID exit=$status end=$(date -Is)"
exit $status
