#!/bin/bash
source ~/.bashrc
source activate mvsenv
realpath `which python`

# Set the PYTHONPATH environment variable to include the parent directory
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
parent_dir="$(dirname "$(dirname "$current_dir")")"
export PYTHONPATH="$parent_dir"
echo $PYTHONPATH

project_path='/raid/sonali/project_mvs/'
submission_id=$1
which_gpu=$2

python -m codebase.inference.inference_wsi --which_cluster="${which_gpu}" --submission_id="${submission_id}" --set="external_test" --epoch='350K' --batch_size=16 --wsi_paths='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/HE_new_wsi/'

#submission_path="${project_path}/results/${submission_id}/"
#chgrp -R mvsgroup "$submission_path"
#chmod -R 770 "$submission_path"

# pfggdx8b_dgm4h_5GP+ASP_selected_snr_dgm4h 350K