#!/bin/bash
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=15G
#SBATCH --time=120:00:00

submission_id="$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)"
echo ${submission_id}
source /cluster/customapps/biomed/grlab/users/bochen/miniconda3/etc/profile.d/conda.sh
conda deactivate
source ~/.bashrc
conda activate multivstain
realpath `which python`

# # Set the PYTHONPATH environment variable to include the parent directory
# current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# parent_dir="$(dirname "$(dirname "$current_dir")")"
# export PYTHONPATH="$parent_dir"
# echo $PYTHONPATH

cv_split="split3"

project_path='/cluster/work/grlab/projects/projects2021-multivstain/'
input_dir_path='/cluster/work/grlab/projects/projects2021-multivstain/data/tupro/'
mkdir /cluster/work/grlab/projects/projects2021-multivstain/results/${submission_id}
chmod -R 770 /cluster/work/grlab/projects/projects2021-multivstain/results/${submission_id}

python -m codebase.experiments.cgan3.main --cv_split="$cv_split" --protein_set="selected_snr" --submission_id="${submission_id}" --no_zip --model_depth=6 --data_path="$input_dir_path" --project_path="$project_path" --imc_prep_seq="raw_clip99_arc_otsu3" --standardize_imc=True --scale01_imc=True --seed=3 --which_cluster="biomed" --which_HE='new' --trans_weights_init='no_init' --dis_weights_init='no_init' --p_flip_jitter_hed_affine='0.5,0,0,0' --which_translator='no_checkerboard' --trans_lr_scheduler='fixedqeual' --dis_lr_scheduler='fixedqeual' --batch_size=16 --n_step=250000 --vis_interval=5000 --multiscale_L1=True --weight_L1=10 