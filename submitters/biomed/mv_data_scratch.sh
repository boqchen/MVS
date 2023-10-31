#!/bin/bash
#SBATCH -p gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=6:00:00
echo ${submission_id}
source /cluster/customapps/biomed/grlab/users/bochen/miniconda3/etc/profile.d/conda.sh
conda deactivate
source ~/.bashrc
conda activate multivstain
realpath `which python`

cv_split="split3"

gpu_assigned=$(squeue --format='%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R' | grep $USER | grep $submission_id | awk '{print $9}' | cut -d"-" -f3)
job_id=$(squeue --format='%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R' | grep $USER | grep $submission_id | awk '{print $1}')
scratch_dir_name="/scratch/slurm-job."$job_id"/mvs_tmp/"
gpu_match_in_whitelist=$(sinfo --Node -o "%N %e %d" | awk '$3>0{print $0}' | awk '$2>300000{print $0}' | awk '{print $1}' | cut -d"-" -f3 | grep $gpu_assigned)

flag_scratch='storage'
input_dir_path='/cluster/work/grlab/projects/projects2021-multivstain/data/tupro/'
### copying data from storage
start_time=$(date +%s)
if [[ $gpu_match_in_whitelist == $gpu_assigned ]]; then
        ### get the directory created for the specific job

	mkdir $scratch_dir_name
        chgrp -R INFK-Raetsch-tp-raetschlab $scratch_dir_name
        cp -r /cluster/work/grlab/projects/projects2021-multivstain/data/tupro/binary_imc_rois_raw_clip99_arc_otsu3_std_minmax_"$cv_split" $scratch_dir_name
        cp -r /cluster/work/grlab/projects/projects2021-multivstain/data/tupro/binary_he_rois $scratch_dir_name

        ### checking whether all data has been copied
        n_imc=340
        n_he=340
        n_imc_copied=$(ls $scratch_dir_name"binary_imc_rois_raw_clip99_arc_otsu3_std_minmax_""$cv_split" | wc -l)
        n_he_copied=$(ls $scratch_dir_name"binary_he_rois" | wc -l)

        if [ "$n_imc" -eq "$n_imc_copied" ] && [ "$n_he" -eq "$n_he_copied" ]; then
                echo "Copied all ROIs"
                flag_scratch='scratch'
		input_dir_path=$scratch_dir_name
        fi
        echo "$gpu_assigned"__"$scratch_dir_name"__"$n_imc_copied"__"$n_he_copied" > "/cluster/work/grlab/projects/projects2021-multivstain/results/""$submission_id"".out"
else
        echo "$gpu_assigned"__"$input_dir_path" > "/cluster/work/grlab/projects/projects2021-multivstain/results/""$submission_id"".out"
fi
end_time=$(date +%s)
run_time=$((end_time - start_time))
echo "Running time: $run_time seconds" >> "/cluster/work/grlab/projects/projects2021-multivstain/results/""$submission_id"".out"