#!/bin/bash
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=4:00:00

source ~/.bashrc
conda deactivate
conda activate mvsenv
realpath `which python`

# Set the PYTHONPATH environment variable to include the parent directory
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
parent_dir="$(dirname "$(dirname "$current_dir")")"
export PYTHONPATH="$parent_dir"
echo $PYTHONPATH

project_path="/cluster/work/grlab/projects/projects2021-multivstain"
which_gpu="biomed"

# get submission_id corresponding to the given job_id 
submission_id=$(ls "${project_path}/results/" | grep "${job_id}")

python -m codebase.eval.select_best_checkpoint --data_set="valid" --project_path="$project_path" --which_cluster="${which_gpu}" --start_epoch=3 --every_x_epoch=3 --submission_id="${submission_id}" --level=2 --avg_kernel=32 --avg_stride=1 --which_HE="new"
# inference on validation set
python -m codebase.inference.inference_roi --set="valid" --epoch="best" --submission_id="${submission_id}" --project_path="$project_path" --which_cluster="${which_gpu}"
# inference on test set
python -m codebase.inference.inference_roi --set="test" --epoch="best" --submission_id="${submission_id}" --project_path="$project_path" --which_cluster="${which_gpu}"


