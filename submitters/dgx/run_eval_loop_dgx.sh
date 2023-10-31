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
res_level=$5
eval_metrics='pcorr,dice,overlap,perc_pos' #'stats,pcorr,ssim,cwssim,psnr,dice,overlap,perc_pos,hmi'

# get the last epoch
last_epoch="$(ls -trlh ${project_path}/results/${submission_id}/tb_logs/ | grep translator | tail -1 | awk '{print $NF}')"
last_epoch=(${last_epoch//_/ })
last_epoch=$(echo ${last_epoch[2]})
last_epoch=(${last_epoch//-/ })
last_epoch=$(echo ${last_epoch[0]})

for i in `seq 10 ${every_x_epoch} $((${last_epoch}))`;
do python -m codebase.eval.eval_protein --submission_id="${submission_id}" --project_path="$project_path" --which_cluster="${which_gpu}" --data_set="${which_set}" --eval_metrics=${eval_metrics} --blur_sigma=1 --kernel_width=3 --level=$res_level --epoch="${i}-1";
done;

# make sure eval metrics on last epoch are also computed
python -m codebase.eval.eval_protein --submission_id="${submission_id}" --project_path="$project_path" --which_cluster="${which_gpu}" --data_set="${which_set}" --eval_metrics=${eval_metrics} --blur_sigma=1 --kernel_width=3 --level=$res_level --epoch="${last_epoch}-1"

submission_path="${project_path}/results/${submission_id}/${which_set}_eval"
chgrp -R mvsgroup "$submission_path"
chmod -R 770 "$submission_path"

# example usage (to generate eval metrics every 10th epoch on level_2 resolution, see last 2 arguments): 
# taskset -c 128-191 bash run_eval_loop_dgx.sh "a9tqnef2_split3_cd3_otsu3_seed3_noinit" "valid" "dgx:5" 10 2