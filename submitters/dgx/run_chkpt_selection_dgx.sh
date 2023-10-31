#!/bin/bash
source ~/.bashrc
source activate mvsenv
realpath `which python`

submission_id=$1
project_path='/raid/sonali/project_mvs/'
data_set=$2
which_gpu=$3

# Set the PYTHONPATH environment variable to include the parent directory
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
parent_dir="$(dirname "$(dirname "$current_dir")")"
export PYTHONPATH="$parent_dir"
echo $PYTHONPATH

python -m codebase.eval.select_best_checkpoint --project_path="$project_path" --which_cluster="${which_gpu}" --start_epoch=3 --every_x_epoch=1 --submission_id="${submission_id}" --level=2 --avg_kernel=32 --avg_stride=1 --which_HE="new" --data_set=${data_set}


