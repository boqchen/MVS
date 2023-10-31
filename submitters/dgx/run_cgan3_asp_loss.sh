#!/bin/bash
experiment_name="dgm4h_5GP+ASP_${2}"
submission_id="$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)_$experiment_name"
echo ${submission_id}
source ~/.bashrc
source activate mvsenv
realpath `which python`

project_path='/raid/sonali/project_mvs/'
results_path=/raid/sonali/project_mvs/results/${submission_id}

mkdir $results_path
chgrp -R mvsgroup $results_path
chmod -R 770 $results_path

# Set the PYTHONPATH environment variable to include the parent directory
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
parent_dir="$(dirname "$(dirname "$current_dir")")"
export PYTHONPATH="$parent_dir"
echo $PYTHONPATH

python -m codebase.experiments.cgan3.main --cv_split="split3" --protein_set="$2" --submission_id="${submission_id}" --no_zip --model_depth=6 --project_path="$project_path" --imc_prep_seq="raw_clip99_arc_otsu3" --standardize_imc=True --scale01_imc=True --seed=3 --which_cluster="$1" --which_HE='new' --trans_weights_init='no_init' --dis_weights_init='no_init' --p_flip_jitter_hed_affine='0.5,0,0,0' --which_translator='no_checkerboard' --trans_lr_scheduler='fixedqeual' --dis_lr_scheduler='fixedqeual' --batch_size=16 --n_step=500000 --vis_interval=5000 --weight_L1=1 --enable_RLW=False --weight_ASP=1 --multiscale_L1=True --weight_multiscale_L1=5 --multiscale=True