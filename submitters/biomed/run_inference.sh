#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=6000
#SBATCH --time=1:00:00
#SBATCH --job-name="inference-mvs"
#SBATCH --output="/cluster/work/grlab/projects/projects2021-multivstain/slurm_cpu_logs/cpu_job-mp%j.out"

source ~/.bashrc
conda activate mvsenv
realpath `which python`

# Set the PYTHONPATH environment variable to include the parent directory
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
parent_dir="$(dirname "$(dirname "$current_dir")")"
export PYTHONPATH="$parent_dir"
echo $PYTHONPATH

submission_id=$1
project_path='/cluster/work/grlab/projects/projects2021-multivstain'
which_set="valid"
which_gpu="cuda:0"


python -m codebase.experiments.inference_roi --set="${which_set}" --epoch='6-4' --submission_id="${submission_id}" --project_path="$project_path" --which_cluster="${which_gpu}"
