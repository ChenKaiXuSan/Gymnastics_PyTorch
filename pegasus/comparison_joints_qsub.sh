#!/bin/bash
# Re-score one method on the comparison joint set (fusion.external.published).
#
#   qsub -v CMD="canonpose --dataset gymnastics --stage evaluate --mode canonical_average --joints comparison12" pegasus/comparison_joints_qsub.sh
#
# CMD   arguments after `python -m fusion external-published` (required; use "::" for spaces if
#       qsub mangles them, this script also accepts a plain quoted string)
# REPO  checkout to run (default: the main working tree)
#
#PBS -A SKIING
#PBS -q gpu
#PBS -l gpunum_job=1
#PBS -l elapstim_req=03:00:00
#PBS -b 1
#PBS -N cmp_joints
#PBS -j o

set -u
REPO="${REPO:-/work/HP260146/chenkaixu/Gymnastics_PyTorch}"
cd "$REPO" || exit 1
: "${CMD:?CMD must be passed via qsub -v}"
export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GYMNASTICS_DATA_ROOT=/work/HP260146/chenkaixu/gymnastics
ENVBIN=/home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body/bin
export PATH="$ENVBIN:$PATH"
echo "[cmp_joints] host=$(hostname) cmd=$CMD start=$(date -Is)"
"$ENVBIN/python" -m fusion external-published ${CMD//::/ }
status=$?
echo "[cmp_joints] exit=$status end=$(date -Is)"
exit $status
