#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000
#SBATCH --time=5:00:00
#SBATCH --job-name="minmax-split3-otsu1-300"
#SBATCH --output="/cluster/work/grlab/projects/projects2021-multivstain/slurm_cpu_logs/cpu_job-mp%j.out"

source ~/.bashrc
conda deactivate
conda activate mvsenv
realpath `which python`

# Set the PYTHONPATH environment variable to include the parent directory
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
parent_dir="$(dirname "$current_dir")"
export PYTHONPATH="$parent_dir"
echo $PYTHONPATH

echo 'Launching imc_std-minmax'
project_path='/cluster/work/grlab/projects/projects2021-multivstain'

python -m codebase.data_preprocess.imc.imc_std-minmax --cv_split='split3' --aligns_set='all' --index_from=300 --last_batch=True --imc_prep_seq='raw_clip99_arc_otsu1' --project_path=${project_path}
# python -m imc_std-minmax.py --cv_split='split1' --aligns_set='valid' --last_batch=True 

# python -m imc_std-minmax.py --cv_split='split1' --aligns_set='train' --index_from=0 --index_to=50 
# python -m imc_std-minmax.py --cv_split='split1' --aligns_set='train' --index_from=50 --index_to=100 
# python -m imc_std-minmax.py --cv_split='split1' --aligns_set='train' --index_from=100 --index_to=150 
# python -m imc_std-minmax.py --cv_split='split1' --aligns_set='train' --index_from=150 --index_to=200 
# python -m imc_std-minmax.py --cv_split='split1' --aligns_set='train' --index_from=200 --last_batch=True 
