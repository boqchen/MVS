#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=10000
#SBATCH --time=4:00:00
#SBATCH --job-name="mvs-mask-ct_basic"
#SBATCH --output="/cluster/work/grlab/projects/projects2021-multivstain/slurm_cpu_logs/cpu_job-mp%j.out"
source ~/.bashrc
conda deactivate
conda activate mvsenv

project_path='/cluster/work/grlab/projects/projects2021-multivstain/'

# Set the PYTHONPATH environment variable to include the parent directory
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
parent_dir="$(dirname "$current_dir")"
export PYTHONPATH="$parent_dir"
echo $PYTHONPATH

python -m codebase.downstream_tasks.cell_typing.get_celltype_masks --project_path=${project_path} --input_path='data/tupro/binary_imc_rois_raw/' --cts_col='ct_basic' --radius=5



