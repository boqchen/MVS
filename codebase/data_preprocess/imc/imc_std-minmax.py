import numpy as np
import pandas as pd
import os
import torch
import random

import torch
from torch.utils.data import Dataset, DataLoader

from codebase.utils.constants import *
from codebase.utils.raw_utils import *
from codebase.experiments.cgan3.training_helpers import *
import matplotlib.pyplot as plt
import time 


# arguments 

parser = argparse.ArgumentParser(description='Data std')
parser.add_argument('--project_path', type=str, required=False, default='/raid/sonali/project_mvs/', help='path where all data, results etc for project reside')
parser.add_argument('--cv_split', type=str, required=False, default='split3', help='cv split for which need to std and minmax data')
parser.add_argument('--aligns_set', type=str, required=True, default='all', help='train/test/valid')
parser.add_argument('--imc_prep_seq', type=str, required=False, default='raw_median_arc', help='which imc normalised data')
parser.add_argument('--index_from', type=int, required=False, default=0)
parser.add_argument('--index_to', type=int, required=False, default=-1)
parser.add_argument('--last_batch', type=str2bool, required=False, default=False)
parser.add_argument('--standardize_imc', type=str2bool, required=False, default=True)
parser.add_argument('--scale01_imc', type=str2bool, required=False, default=True)

args = parser.parse_args()

cv_split = args.cv_split
aligns_set = args.aligns_set
imc_prep_seq = args.imc_prep_seq

good_areas_only = False
precise_only = False
standardize_imc = args.standardize_imc #True 
scale01_imc = args.scale01_imc #True

DATA_DIR = os.path.join(args.project_path,'data/tupro')
trafos = '_'
if standardize_imc:
    trafos=trafos+'std_'
if scale01_imc:
    trafos = trafos+'minmax_'
save_path = os.path.join(DATA_DIR, 'binary_imc_rois_'+imc_prep_seq+trafos+cv_split)
if not os.path.exists(save_path): 
    os.makedirs(save_path)
    
# getting cohort stasts     
if standardize_imc:
    # load cohort stats based on imc preprocessing steps (naming convention)
    cohort_stats_name = 'imc_rois_'+imc_prep_seq+'-agg_stats.tsv'
    cohort_stats = pd.read_csv(os.path.join(args.project_path,COHORT_STATS_PATH,cv_split,cohort_stats_name), sep='\t', index_col=[0])
     
    avg_mat = cohort_stats['mean_cohort']
    std_mat = cohort_stats['std_cohort']

if scale01_imc:
    min_col = 'min_stand_cohort' if standardize_imc else 'min_cohort'
    max_col = 'max_stand_cohort' if standardize_imc else 'max_cohort'
    min_mat = cohort_stats[min_col]
    max_mat = cohort_stats[max_col]

    
# running for each split + saving 
cv = json.load(open(Path(args.project_path).joinpath(CV_SPLIT_ROIS_PATH)))
if aligns_set == 'all':
    align_results = cv[cv_split]['train']
    align_results.extend(cv[cv_split]['valid'])
    align_results.extend(cv[cv_split]['test'])
else:
    align_results = cv[cv_split][aligns_set]
    
print('aligns_set: ', aligns_set)

if args.last_batch:
    samples = align_results[args.index_from:]
else: 
    samples = align_results[args.index_from:args.index_to]
    
for ar_idx, ar in enumerate(samples):
    save_path_roi = os.path.join(save_path, ar + ".npy")
    if not os.path.exists(save_path_roi): 

        IMC_ROI_STORAGE = get_imc_roi_storage(DATA_DIR, imc_prep_seq, False, False, cv_split)
        imc_roi = np.load(os.path.join(IMC_ROI_STORAGE, ar + ".npy"))

        print(ar, imc_roi.shape, ar_idx)

        if standardize_imc:
            def standardize(x, mean_cohort, std_cohort):
                return (x-mean_cohort)/std_cohort
            imc_roi = np.apply_along_axis(standardize, 2, imc_roi, mean_cohort=avg_mat, std_cohort=std_mat)

        if scale01_imc:
            def min_max_scale(x, min_cohort, max_cohort):
                return (x-min_cohort)/(max_cohort - min_cohort)
            imc_roi = np.apply_along_axis(min_max_scale, 2, imc_roi, min_cohort=min_mat, max_cohort=max_mat)

        np.save(save_path_roi, imc_roi)
