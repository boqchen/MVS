#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000
#SBATCH --time=24:00:00
#SBATCH --job-name="write_imc_raw-set1"
source ~/.bashrc
source activate mvsenv2
realpath `which python`

# Set the PYTHONPATH environment variable to include the parent directory
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
parent_dir="$(dirname "$current_dir")"
export PYTHONPATH="$parent_dir"
echo $PYTHONPATH

python -m codebase.data_preprocess.imc.write_imc_np --save_path='/cluster/work/grlab/projects/projects2021-multivstain/data/tupro/binary_imc_rois_raw/' --protein_list="PROTEIN_LIST_MVS" --sample_list='MACOLUD,MADAJEJ,MADEGOD,MADUBIP,MADUFEM,MAFIBAF,MAHEFOG,MAHOBAM,MAJEFEV,MAJOFIJ,MAKYGIW,MANOFYB'
