#!/bin/bash
# Submit the SportsPose SAM3D inference as one gpu job per group of subjects.
#
#   bash pegasus/submit_sportspose_infer.sh            # 26 subjects in 7 jobs
#   PER_JOB=2 bash pegasus/submit_sportspose_infer.sh  # smaller jobs
set -eu
PER_JOB="${PER_JOB:-4}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p local/runs/sportspose_benchmark/joblogs
submit() {  # day, subjects...
  local day="$1"; shift
  local subjects="$*"
  local tag="${day}_$(echo "$subjects" | tr ' ' '-' )"
  local log="local/runs/sportspose_benchmark/joblogs/infer_${tag}.log"
  echo "qsub -N sp_${day:0:2}_${1} -o $log -v REPO=$REPO,DAYS=$day,SUBJECTS=\"$subjects\" pegasus/benchmark_sportspose_infer_qsub.sh"
  qsub -N "sp_${day:0:2}_${1}" -o "$log" -v "REPO=$REPO,DAYS=$day,SUBJECTS=$subjects" pegasus/benchmark_sportspose_infer_qsub.sh
}
for day in indoors outdoors; do
  mapfile -t subjects < <(ls "${GYMNASTICS_PUBLIC_DATASETS_ROOT:-/work/HP260146/chenkaixu/public_datasets}/multiview_human/SportsPose/data/$day" | grep '^S[0-9][0-9]$')
  for ((i = 0; i < ${#subjects[@]}; i += PER_JOB)); do
    submit "$day" "${subjects[@]:i:PER_JOB}"
  done
done
