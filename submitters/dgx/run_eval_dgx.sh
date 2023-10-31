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
which_gpu=$2
res_level=$3
eval_metrics='pcorr,cwssim' #'stats,pcorr,ssim,cwssim,psnr,dice,overlap,perc_pos,hmi'

sets=('test' 'valid')

for set in "${sets[@]}"; do
    # on best epoch
    python -m codebase.eval.eval_protein --submission_id="${submission_id}" --project_path="$project_path" --which_cluster="${which_gpu}" --data_set="${set}" --eval_metrics=${eval_metrics} --blur_sigma=1 --kernel_width=3 --level=$res_level --epoch="best"
    # on last epoch 
    python -m codebase.eval.eval_protein --submission_id="${submission_id}" --project_path="$project_path" --which_cluster="${which_gpu}" --data_set="${set}" --eval_metrics=${eval_metrics} --blur_sigma=1 --kernel_width=3 --level=$res_level --epoch="last"
done

submission_path="${project_path}/results/${submission_id}/${which_set}_eval"
chgrp -R mvsgroup "$submission_path"
chmod -R 770 "$submission_path"

# example usage (to generate eval metrics every 10th epoch on level_2 resolution, see last 2 arguments): 
# taskset -c 128-191 bash run_eval_dgx.sh "a9tqnef2_split3_cd3_otsu3_seed3_noinit" "dgx:5" 2