import numpy as np
import pandas as pd
import json
import math
import os
import argparse
import cv2
import tifffile

from codebase.utils.constants import *
from codebase.utils.raw_utils import *
from codebase.utils.dataset_utils import get_patient_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Write IMC as numpy arrays.')
    parser.add_argument('--save_path', type=str, required=False, default='/cluster/work/grlab/projects/projects2021-multivstain/data/tupro/', help='Path to save the processed data')
    parser.add_argument('--sample_list', type=str, required=False, default='', help='List of samples to parse, separated by comma.')
    parser.add_argument('--overwrite', type=str2bool, required=False, default=False, help='Whether to overwrite if output files exist')
    parser.add_argument('--protein_list_name', type=str, required=False, default="PROTEIN_LIST_MVS", help='Name of the protein list to use {PROTEIN_LIST, PROTEIN_LIST_FULL, PROTEIN_LIST_FULL_IR}, consult code.utils.constants.py')
    
    args = parser.parse_args()
    # protein list
    protein_list = get_protein_list_by_name(args.protein_list_name)

    # sample list
    if args.sample_list == '':
        sel_samples = MVS_SAMPLES
    else:
        sel_samples = args.sample_list.split(',')
    print('#samples: '+str(len(sel_samples)))
    
    sample_patient_df = get_patient_samples(data_dir = '/cluster/work/tumorp/data_repository/',
                                            phase='study', indication='melanoma')
    sample_patient_dict = dict(zip(sample_patient_df['sample'], sample_patient_df['patient']))
    
    for i,s in enumerate(sel_samples):
        print(i,s)
        patient = sample_patient_dict[s]
        
        imc_raw_folder = os.path.join(ROOT_DIR_STUDY, patient, s, "imc/raw/")
        # add latest pass subdir
        pass_dir = sorted(os.listdir(imc_raw_folder))[-1]
        imc_raw_folder = os.path.join(imc_raw_folder, pass_dir)
        imc_raw_txt_candidates = [f.name for f in os.scandir(imc_raw_folder) if f.is_file() and f.name.endswith(".txt")]
        imc_raw_txt_candidates = [x for x in imc_raw_txt_candidates if '__' in x]
        #imc_raw_txt_candidates = [roi for roi in imc_raw_txt_candidates if roi.split('__')[1].split('_')[0] not in REF_ROIS]
        # for some samples the naming convention was different, so had to add a special case
        if s in ['MEKOBAB', 'MEVIXYV', 'MISYPUP', 'MOBUBOT', 'MOBYLUD', 'MOMUSIG', 'MOPYPUS', 'MOVAZYQ']:
            imc_raw_txt_candidates = [roi for roi in imc_raw_txt_candidates if roi.split('__')[1].split('_')[1] in IMC_ROIS]
            rois = sorted([roi.split('__')[1].split('_')[1] for roi in imc_raw_txt_candidates])
        else:
            imc_raw_txt_candidates = [roi for roi in imc_raw_txt_candidates if roi.split('__')[1].split('_')[0] in IMC_ROIS]
            rois = sorted([roi.split('__')[1].split('_')[0] for roi in imc_raw_txt_candidates])
        print('Found '+str(len(rois))+' ROIs')
        
        for roi_id in rois:
            save_file = os.path.join(args.save_path, s + "_" + roi_id + ".npy")
            roi_candidates = [os.path.join(imc_raw_folder, irtc) for irtc in imc_raw_txt_candidates if roi_id in irtc[irtc.rfind("__"):]]

            if len(roi_candidates) != 1:
                if len(roi_candidates) > 1:
                    # for some samples there are multiple files per ROI name,
                    # eg due to references which have longer file names -> taking file with min name length
                    roi_candidates = [roi_candidates[np.argmin([len(x) for i,x in enumerate(roi_candidates)])]]
                else:
                    print("Problematic to find corresponding raw IMC file for sample ", s, " and ROI ", roi_id)
                    continue  # continue on ROI ID loop
                    
            if not args.overwrite and os.path.exists(save_file):
                print('Data for sample '+s+' and roi '+roi_id+' already exist -> skipping')
                continue

            imc_raw_txt_file = roi_candidates[0]
            try:
                imc_roi_np = read_raw_protein_txt(imc_raw_txt_file, transformation=(lambda x: x), protein_subset=protein_list)
            except:
                print("Couldn't compute for sample "+s+' for ROI '+roi_id)
                continue
                
            if not args.overwrite and os.path.exists(save_file):
                continue
            print(save_file)
            np.save(save_file, imc_roi_np)
                