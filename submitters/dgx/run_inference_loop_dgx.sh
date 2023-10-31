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
every_x_epoch=$4

# get the last epoch
last_epoch="$(ls -trlh ${project_path}/results/${submission_id}/tb_logs/ | grep translator | tail -1 | awk '{print $NF}')"
last_epoch=(${last_epoch//_/ })
last_epoch=$(echo ${last_epoch[2]})
last_epoch=(${last_epoch//-/ })
last_epoch=$(echo ${last_epoch[0]})

for i in `seq ${every_x_epoch} ${every_x_epoch} $((${last_epoch}))`;
do python -m codebase.experiments.inference_roi --set="${which_set}" --epoch="${i}-1" --submission_id="${submission_id}" --project_path="$project_path" --which_cluster="${which_gpu}";
done;

# make sure inference on last epoch is also computed
python -m codebase.inference.inference_roi --set="${which_set}" --epoch="last" --submission_id="${submission_id}" --project_path="$project_path" --which_cluster="${which_gpu}"

submission_path="${project_path}/results/${submission_id}/${which_set}_images"
chgrp -R mvsgroup "$submission_path"
chmod -R 770 "$submission_path"

# example usage (to generate images every 10th epoch, see last argument): 
# bash run_inference_loop_dgx.sh "a9tqnef2_split3_cd3_otsu3_seed3_noinit" "valid" "dgx:5" 10

