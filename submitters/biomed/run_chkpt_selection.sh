#!/bin/bash
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=2:00:00

source ~/.bashrc
conda deactivate
conda activate mvsenv
realpath `which python`

# Set the PYTHONPATH environment variable to include the parent directory
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
parent_dir="$(dirname "$(dirname "$current_dir")")"
export PYTHONPATH="$parent_dir"
echo $PYTHONPATH

submission_id=$1
project_path='/cluster/work/grlab/projects/projects2021-multivstain'
data_set=$2
which_gpu="biomed" #"cuda:0"

python -m codebase.eval.select_best_checkpoint --project_path="$project_path" --which_cluster="${which_gpu}" --start_epoch=1 --every_x_epoch=3 --submission_id="${submission_id}" --level=2 --avg_kernel=75 --avg_stride=1 --which_HE='new' --data_set=${data_set}


