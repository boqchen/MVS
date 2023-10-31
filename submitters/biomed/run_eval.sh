#!/bin/bash
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=25G
#SBATCH --time=4:00:00
#SBATCH --job-name="eval-mvs"
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

for res_level in 2 4 6
do
python -m codebase.eval.eval_protein --submission_id="${submission_id}" --data_set="${which_set}" --epoch='last' --eval_metrics='pcorr,ssim,cwssim' --blur_sigma=3 --level=$res_level --which_cluster="${which_gpu}" --project_path="$project_path"
done
