#!/bin/bash
source ~/.bashrc
source activate mvsenv
realpath `which python`

# Set the PYTHONPATH environment variable to include the parent directory
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
parent_dir="$(dirname "$(dirname "$current_dir")")"
export PYTHONPATH="$parent_dir"
echo $PYTHONPATH

submission_id=$1
project_path='/raid/sonali/project_mvs/'
which_set=$2
which_gpu=$3
which_epoch=$4

python -m codebase.inference.inference_roi --set="${which_set}" --epoch="${which_epoch}" --submission_id="${submission_id}" --project_path="$project_path" --which_cluster="${which_gpu}"

submission_path="${project_path}/results/${submission_id}/${which_set}_images"
chgrp -R mvsgroup "$submission_path"
chmod -R 770 "$submission_path"
