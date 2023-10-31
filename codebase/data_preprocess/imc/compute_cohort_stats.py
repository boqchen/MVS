import numpy as np
import pandas as pd
import json
import math
import os
import argparse
from pathlib import Path

from codebase.utils.constants import *
#from code.utils.raw_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute cohort stats.')
    parser.add_argument('--project_path', type=str, required=False, default='/raid/sonali/project_mvs/', help='path where all data, results etc for project reside')
    parser.add_argument('--input_path', type=str, required=False, default='/cluster/work/grlab/projects/projects2021-multivstain/data/tupro/binary_imc_rois_raw/', help='Path to with input data')
    parser.add_argument('--save_path', type=str, required=False, default='/cluster/work/grlab/projects/projects2021-multivstain/data/tupro/agg_stats_qc/', help='Path to save the processed data')
    parser.add_argument('--save_fname', type=str, required=False, default='tmp')
    parser.add_argument('--cv_split', type=str, required=False, default='mvs_cohort', help='CV split name to compute cohort stats on training set only; if mvs_cohort, then all data will be used for stats computation')
    
    args = parser.parse_args()
    
    protein_list = PROTEIN_LIST
    # for the newly processed data need to use the updated protein list
    if ((args.input_path.split('/')[-1] not in ['binary_imc_rois','binary_imc_rois_simon']) and (args.input_path.split('/')[-2] not in ['binary_imc_rois', 'binary_imc_rois_simon'])):
        protein_list = PROTEIN_LIST_MVS
    
    def minmax_cohort_scale(ar, min_cohort, max_cohort):
        ''' Function to min-max scale numpy array using cohort stats
        ar: numpy array with proteins in axis 2
        min_cohort: np.array of length #proteins with min per protein
        max_cohort: np.array of length #proteins with max per protein
        '''
        return (ar - min_cohort)/ (max_cohort - min_cohort)

#### The commented code is more elegant but does not work due to very high memory requirement
#     if args.cv_split=='mvs_cohort':
#         sel_samples = MVS_SAMPLES
#     else:
#         cv_df = json.load(open(os.path.join(args.project_path,CV_SPLIT_SAMPLES_PATH)))
#         sel_samples = cv_df[args.cv_split]['train']

#     files = os.listdir(args.input_path)
#     files = [x for x in files if x.split('_')[0] in sel_samples]
#     for i,fname in enumerate(files):
#         if i % 10 == 0:
#             print(i)
#         df = np.load(os.path.join(args.input_path, fname))
#         # for Simon's files there were 38 proteins (MPO and CD15 were removed), thus creating initial matrix here to adjust
#         if i == 0:
#             df_all = np.empty((1, min(df.shape[2], len(protein_list))))
#         try:
#             df_all = np.vstack([df_all, df.reshape(df.shape[0]*df.shape[1], df.shape[2])])
#         except ValueError:
#             print('missing proteins!')
#             print(i, fname)
#             import pdb; pdb.set_trace()
#             continue

#     mean_cohort = df_all.mean(axis=0)
#     std_cohort = df_all.std(axis=0)
#     min_cohort = df_all.min(axis=0)
#     max_cohort = df_all.max(axis=0)

#     # compute stats on minmaxed values
#     df_all_minmaxed = minmax_cohort_scale(df_all, min_cohort, max_cohort)
#     mean_minmaxed_cohort = df_all_minmaxed.mean(axis=0)
#     std_minmaxed_cohort = df_all_minmaxed.std(axis=0)

    files = os.listdir(args.input_path)
    if args.cv_split=='mvs_cohort':
        sel_samples = MVS_SAMPLES
        files = [x for x in files if x.split('_')[0] in sel_samples]
    else:
        cv_df = json.load(open(os.path.join(args.project_path,CV_SPLIT_ROIS_PATH)))
        sel_samples = cv_df[args.cv_split]['train']
        files = [x for x in files if x.split('.')[0] in sel_samples]

    min_cohort = np.zeros(len(protein_list))
    max_cohort = np.zeros(len(protein_list))
    sum_cohort = np.zeros(len(protein_list))
     
    
    n_cohort = 0
    for i,fname in enumerate(files):
        if i % 10 == 0:
            print(i)
        df = np.load(os.path.join(args.input_path, fname))
        min_cohort = np.min(np.vstack([min_cohort, np.min(np.min(df, axis=1),axis=0)]), axis=0)
        max_cohort = np.max(np.vstack([max_cohort, np.max(np.max(df, axis=1),axis=0)]), axis=0)
        sum_cohort = np.sum([sum_cohort, np.sum(np.sum(df, axis=1),axis=0)], axis=0)
        n_cohort = n_cohort+(df.shape[0]*df.shape[1])

    mean_cohort = sum_cohort/n_cohort
        
    # need to load all data again to calculate std dev on input values and mean on minmaxed values
    ssq_cohort = np.zeros(len(protein_list))
    sum_minmaxed_cohort = np.zeros(len(protein_list))
    for i,fname in enumerate(files):
        if i % 10 == 0:
            print(i)
        df = np.load(os.path.join(args.input_path, fname))
        # calculate sum of squares on original values
        ssq = np.sum((df.reshape(df.shape[0]*df.shape[1], df.shape[2]) - mean_cohort)**2, axis=0)
        ssq_cohort = np.sum([ssq_cohort, ssq], axis=0)
        # sum of minmaxed values
        df_minmaxed = minmax_cohort_scale(df, min_cohort, max_cohort)
        sum_minmaxed_cohort = np.sum([sum_minmaxed_cohort, np.sum(np.sum(df_minmaxed, axis=1),axis=0)], axis=0)

    std_cohort = np.sqrt(ssq_cohort/n_cohort)
    mean_minmaxed_cohort = sum_minmaxed_cohort/n_cohort

    # need to load all data again to calculate std dev on minmaxed values and min, max on standardized
    ssq_minmaxed_cohort = np.zeros(len(protein_list))
    for i,fname in enumerate(files):
        if i % 10 == 0:
            print(i)
        df = np.load(os.path.join(args.input_path, fname))
        # calculate sum of squares on minmaxed values
        df_minmaxed = minmax_cohort_scale(df, min_cohort, max_cohort)
        ssq_minmaxed = np.sum((df_minmaxed.reshape(df_minmaxed.shape[0]*df_minmaxed.shape[1], df_minmaxed.shape[2]) - mean_minmaxed_cohort)**2, axis=0)
        ssq_minmaxed_cohort = np.sum([ssq_minmaxed_cohort, ssq_minmaxed], axis=0)

    std_minmaxed_cohort = np.sqrt(ssq_minmaxed_cohort/n_cohort)

    # min, max on standardized data
    min_stand_cohort = (min_cohort - mean_cohort)/std_cohort
    max_stand_cohort = (max_cohort - mean_cohort)/std_cohort
    
    stats_df = pd.DataFrame({'min_cohort':min_cohort, 'max_cohort': max_cohort,
                             'mean_cohort':mean_cohort, 'std_cohort': std_cohort,
                             'min_stand_cohort':min_stand_cohort, 'max_stand_cohort':max_stand_cohort,
                            'mean_minmaxed_cohort':mean_minmaxed_cohort, 'std_minmaxed_cohort':std_minmaxed_cohort},
                            index=protein_list)
    
    save_path = Path(args.save_path).joinpath(args.cv_split)
    stats_df.to_csv(save_path.joinpath(args.save_fname+'-agg_stats.tsv'), sep='\t')
    
