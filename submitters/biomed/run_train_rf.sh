#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000
#SBATCH --time=4:00:00
#SBATCH --job-name="rf-train-snrv2-r5-depth20"
#SBATCH --output="/cluster/work/grlab/projects/projects2021-multivstain/slurm_cpu_logs/cpu_job-mp%j.out"
source ~/.bashrc
conda deactivate
conda activate mvsenv

# Set the PYTHONPATH environment variable to include the parent directory
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
parent_dir="$(dirname "$current_dir")"
export PYTHONPATH="$parent_dir"
echo $PYTHONPATH

project_path='/cluster/work/grlab/projects/projects2021-multivstain/'
protein_set='selected_snr_v2'


echo "rf-train-snrv2-r5-depth20"

python -m codebase.downstream_tasks.cell_typing.train_rf --cv_split='split3' --project_path=${project_path} --protein_list=${protein_set} --input_file_path='/cluster/work/grlab/projects/projects2021-multivstain/data/tupro/imc_updated/agg_masked_data-raw_clip99_arc_otsu3_std_minmax_split3-r5.tsv' --max_depth=20