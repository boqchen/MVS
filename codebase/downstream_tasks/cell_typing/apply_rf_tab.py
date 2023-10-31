import os
from pathlib import Path
import numpy as np
import pandas as pd
import random
import joblib
import json
import argparse
import sklearn
import warnings
warnings.simplefilter("ignore")
from sklearn.ensemble import RandomForestClassifier

from codebase.utils.constants import *
from codebase.utils.eval_utils import get_protein_list
from codebase.utils.raw_utils import str2bool
from codebase.downstream_tasks.cell_typing.utils_rf import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply RF on tabular predicted pseudo-single-cell data.')
    parser.add_argument('--project_path', type=str, required=False, default='/raid/sonali/project_mvs/', help='Path where all data, results etc for project reside')
    parser.add_argument('--input_path', type=str, required=False, default='', help='Path with input data (if not specified, then using submission id etc to determine the path)')
    parser.add_argument('--save_path', type=str, required=False, default='', help='Path to save predictions (if not specified, then using submission id etc to determine the path)')
    parser.add_argument('--rf_fname', type=str, required=False, default='rf-cell_type-selected_snr-raw_clip99_arc_otsu3_std_minmax_split3-r5-ntrees100-maxdepth30.joblib', help='Name fo the trained RF joblib file')
    parser.add_argument('--submission_id', type=str, required=False, default=None, help='Job submission_id')
    parser.add_argument('--epoch', type=str, required=False, default='last', help='Which epoch to use (e.g.,2 or 2-3)')
    parser.add_argument('--level', type=int, required=False, default=2, help='Which resolution to use {2,4,6}')
    parser.add_argument('--cv_split', type=str, required=False, default='', help='Selected CV split, if not specified, then read from job args')
    parser.add_argument('--data_set', type=str, required=False, default="test", help='Which set from split to use {test, valid, train}')
    parser.add_argument('--overwrite', type=str2bool, required=False, default=False, help='Whether to overwrite if output files exist')
    args = parser.parse_args()
    
    args = parser.parse_args()

    
    PROJECT_PATH = Path(args.project_path)
    if args.save_path == '':
        SAVE_PATH = PROJECT_PATH.joinpath('results', args.submission_id, args.data_set+'_ct', 'epoch'+str(args.epoch), 'level_'+str(args.level))
    else:
        SAVE_PATH = Path(args.save_path)
    if not os.path.exists(SAVE_PATH):
        SAVE_PATH.mkdir(parents=True, exist_ok=False)
    # save argument values into a txt file
    with open(Path(SAVE_PATH).joinpath('rf_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    if args.input_path == '':
        INPUT_PATH = PROJECT_PATH.joinpath('results', args.submission_id, args.data_set+'_scdata', 'epoch'+str(args.epoch), 'level_'+str(args.level))
        # Get job args
        job_args = json.load(open(PROJECT_PATH.joinpath('results',args.submission_id, 'args.txt')))
        cv_split = job_args['cv_split']
    else:
        INPUT_PATH = Path(args.input_path)
        cv_split = args.cv_split
   
    # Get sample_roi list for s given split and data_set
    cv = json.load(open(PROJECT_PATH.joinpath(CV_SPLIT_ROIS_PATH)))
    sample_rois = cv[cv_split][args.data_set]
    
    # Load trained RF
    rf = joblib.load(Path(PROJECT_PATH).joinpath('meta','rf',cv_split, args.rf_fname))
    protein_list = get_protein_list(args.rf_fname.split('-')[2])
    sel_cols = protein_list
    sel_cols.extend(['sample_roi','X','Y', 'radius'])
    
    for s_roi in sample_rois:
        # Get protein expression data
        df_roi = pd.read_csv(INPUT_PATH.joinpath(s_roi+'.tsv'), sep='\t', index_col=[0])
        # Remove any rows with NaN and get proteins used for training RF
        print('Removing '+str(sum(df_roi.isna().sum(axis=1)>0))+' objects')
        df_roi = df_roi.loc[~(df_roi.isna().sum(axis=1)>0),sel_cols]
        # apply RF
        assert 'cell_type' in args.rf_fname, 'Wrong cell types used, only default CELL_TYPES supported for now'
        preds = get_manual_aggregation(rf, df_roi.loc[:,~df_roi.columns.isin(['sample_roi','X','Y','radius'])], CELL_TYPES)
        preds = pd.DataFrame(preds, index=df_roi.index, columns=CELL_TYPES)
        # extract cell-type label per cell
        preds = preds.apply(lambda x: [CELL_TYPES[i] for i,y in enumerate(x) if np.isclose(y,1)][0], axis=1).to_frame('pred_cell_type')
        # merge with coords and save
        df_roi = df_roi.loc[:,['X','Y']].merge(preds, left_index=True, right_index=True, how='left')
        df_roi.to_csv(SAVE_PATH.joinpath(s_roi+'.tsv'), sep='\t')
