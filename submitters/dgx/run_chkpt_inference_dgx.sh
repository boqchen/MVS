#!/bin/bash
source ~/.bashrc
source activate mvsenv
realpath `which python`

submission_id=$1
which_gpu=$2
every_x_epoch=$3
project_path='/raid/sonali/project_mvs/'

# Set the PYTHONPATH environment variable to include the parent directory
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
parent_dir="$(dirname "$(dirname "$current_dir")")"
export PYTHONPATH="$parent_dir"
echo $PYTHONPATH

python -m codebase.eval.select_best_checkpoint --data_set="valid" --which_cluster="${which_gpu}" --start_epoch=1 --every_x_epoch=${every_x_epoch} --submission_id="${submission_id}" --level=2 --avg_kernel=64 --avg_stride=1 
# inference on validation set
python -m codebase.inference.inference_roi --set="valid" --epoch="best" --submission_id="${submission_id}" --which_cluster="${which_gpu}"
python -m codebase.inference.inference_roi --set="valid" --epoch="last" --submission_id="${submission_id}" --which_cluster="${which_gpu}"

# inference on test set
python -m codebase.inference.inference_roi --set="test" --epoch="best" --submission_id="${submission_id}" --which_cluster="${which_gpu}"
python -m codebase.inference.inference_roi --set="test" --epoch="last" --submission_id="${submission_id}" --which_cluster="${which_gpu}"

submission_path="${project_path}/results/${submission_id}/chkpt_selection"
chgrp -R mvsgroup "$submission_path"
chmod -R 770 "$submission_path"

submission_path="${project_path}/results/${submission_id}/valid_images"
chgrp -R mvsgroup "$submission_path"
chmod -R 770 "$submission_path"

submission_path="${project_path}/results/${submission_id}/test_images"
chgrp -R mvsgroup "$submission_path"
chmod -R 770 "$submission_path"


