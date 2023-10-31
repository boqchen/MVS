#!/bin/bash
experiment_name="split3_otsu3_seed3_selected_snr"
submission_id="$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)_$experiment_name"
echo ${submission_id}
source ~/.bashrc
source activate mvsenv
realpath `which python`

data_dir_path='/raid/sonali/project_mvs/'
results_path=/raid/sonali/project_mvs/results/${submission_id}
mkdir $results_path
chgrp -R mvsgroup $results_path
chmod -R 770 $results_path

# Set the PYTHONPATH environment variable to include the parent directory
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
parent_dir="$(dirname "$(dirname "$current_dir")")"
export PYTHONPATH="$parent_dir"
echo $PYTHONPATH

python -m codebase.experiments.cgan3.main --epochs=120 --cv_split="split3" --protein_set="selected_snr" --submission_id="${submission_id}" --no_zip --model_depth=6 --imc_prep_seq="raw_clip99_arc_otsu3" --standardize_imc=True --scale01_imc=True --seed=3 --which_cluster=dgx:2 --which_HE='new' --trans_weights_init='no_init' --dis_weights_init='no_init' --p_flip_jitter_hed_affine='0.5,0,0,0' --which_translator='no_checkerboard'
