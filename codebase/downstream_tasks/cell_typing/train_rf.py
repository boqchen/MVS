import os
from pathlib import Path
import numpy as np
import pandas as pd
import random
import joblib
import json
import argparse

import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import confusion_matrix as cfm

from codebase.utils.constants import *
from codebase.utils.eval_utils import get_protein_list
from codebase.downstream_tasks.cell_typing.utils_rf import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RF clasifier using train set.')
    parser.add_argument('--project_path', type=str, required=False, default='/cluster/work/grlab/projects/projects2021-multivstain/', help='Path where all data, results etc for project reside')
    parser.add_argument('--save_path', type=str, required=False, default='meta/rf/', help='Path to save predictions')
    parser.add_argument('--input_file_path', type=str, required=True, default='data/tupro/imc_updated/agg_masked_data-raw_clip99_arc_otsu3-r5.tsv', help='Path starting from project_path to the input file with aggregated protein expression data')
    parser.add_argument('--cv_split', type=str, required=True, default=None, help='Selected CV split, if None then the splitting used for the report is used')
    parser.add_argument('--cts_col', type=str, required=False, default='cell_type', help='Cell-typing level (col name) to use for training RF')
    parser.add_argument('--protein_list', type=str, required=False, default='selected_snr_v2', help='Protein set to use for training RF')
    parser.add_argument('--seed', type=int, required=False, default=0, help='Random seed')
    parser.add_argument('--n_estimators', type=int, required=False, default=100, help='Number of trees to use')
    parser.add_argument('--max_depth', type=int, required=False, default=None, help='Max tree depth (if None, then tree grown until purity, see sklearn docs')
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    protein_list = get_protein_list(args.protein_list)
    PROJECT_PATH = Path(args.project_path)
    save_path = PROJECT_PATH.joinpath(args.save_path, args.cv_split)
    if not os.path.exists(save_path):
        save_path.mkdir(parents=True, exist_ok=False)
    save_fname = '-'.join(['rf', args.cts_col, args.protein_list, 
                           args.input_file_path.split('/')[-1].split('.')[0].replace('agg_masked_data-',''),
                           'ntrees'+str(args.n_estimators), 'maxdepth'+str(args.max_depth)])

    # Load train and valid sample_roi assignments
    cv = json.load(open(PROJECT_PATH.joinpath(CV_SPLIT_ROIS_PATH)))
    train_aligns = cv[args.cv_split]['train']
    valid_aligns = cv[args.cv_split]['valid']
    sets = [train_aligns, valid_aligns]
    set_names = ['train', 'valid']

    # Get protein expression data
    df_all = pd.read_csv(PROJECT_PATH.joinpath(args.input_file_path), sep='\t', index_col=[0])
    # subset to ROIs of interest
    df_all = df_all.loc[df_all.sample_roi.isin(train_aligns+valid_aligns),:]
    # Make sure there are no Nan's (not sure why one cells appears to be nan => check in agg_masks_data)
    df_all = df_all.loc[~df_all.iloc[:,0].isna(),:]
    df_all['set'] = ['train' if x in train_aligns else 'valid' for x in df_all.sample_roi.to_list()]
    
    # Get cell-type labels and merge with protein data
    cts = pd.read_csv(PROJECT_PATH.joinpath('data/tupro/imc_updated/coldata.tsv'), sep='\t', index_col=[0])
    df_all = df_all.merge(cts.loc[:,[args.cts_col]], left_index=True, right_index=True, how='inner')
    df_all = df_all.sort_values(by=['set',args.cts_col])
    CELL_TYPES = sorted(df_all[args.cts_col].unique())
    print('Loaded data')
    
    # One-hot encode cell-type labels
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    # subset to protein set of interest and split into x and y
    train_x = df_all.loc[df_all.set=='train', protein_list]
    train_y = ohe.fit_transform(df_all.loc[df_all.set=='train', args.cts_col].to_numpy().reshape(-1, 1))
    train_y = pd.DataFrame(train_y, columns=CELL_TYPES).astype(int)

    valid_x = df_all.loc[df_all.set=='valid', protein_list]
    valid_y = ohe.transform(df_all.loc[df_all.set=='valid', args.cts_col].to_numpy().reshape(-1, 1))
    valid_y = pd.DataFrame(valid_y, columns=CELL_TYPES).astype(int)

    # Make sure we have a 'one' in every y output vector
    assert np.sum(train_y.sum()) == train_y.shape[0] and np.sum(valid_y.sum()) == valid_y.shape[0]
    print('One-hot encoded cell-type labels')
        
    # Train RF
    rf = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=4, class_weight='balanced', max_depth=args.max_depth, oob_score=True)
    rf.fit(train_x, train_y)
    print('RF trained')
    pd.DataFrame({'oob':rf.oob_score_}, index=[0]).to_csv(save_path.joinpath(save_fname+'-oob.txt'), sep='\t', index=False)
    
    # Feature importance plot
    plot_feature_imp(protein_list, rf.feature_importances_)
    plt.savefig(save_path.joinpath(save_fname+'-fi.png'))
    
    df_all['RF_preds'] = 'other'
    for set_name in set_names:
        preds = get_manual_aggregation(rf, df_all.loc[df_all.set==set_name, protein_list], CELL_TYPES)
        gt = ohe.fit_transform(df_all.loc[df_all.set==set_name, args.cts_col].to_numpy().reshape(-1, 1))
        f1 = f1_score(gt, preds, average="weighted")
        print(set_name.upper()+" F1: ", f1)
        
        df_all.loc[df_all.set==set_name,'RF_preds'] = ohe.inverse_transform(preds)
        class_report = classification_report(df_all.loc[df_all.set==set_name,args.cts_col],df_all.loc[df_all.set==set_name,'RF_preds'],output_dict=True)
        pd.DataFrame(class_report).transpose().to_csv(save_path.joinpath(save_fname+'-report_'+set_name+'.tsv'), sep='\t')
        print(class_report)
        # cfm returns a matrix with GT as index and pred as columns
        confusion_matrix = cfm(df_all.loc[df_all.set==set_name,args.cts_col],df_all.loc[df_all.set==set_name,'RF_preds'])
        confusion_matrix = pd.DataFrame(confusion_matrix, index=CELL_TYPES, columns=CELL_TYPES)
        confusion_matrix.to_csv(save_path.joinpath(save_fname+'-cfm_'+set_name+'.tsv'), sep='\t')
        print(confusion_matrix)
        plot_cfm(df_all.loc[df_all.set==set_name,args.cts_col], df_all.loc[df_all.set==set_name,'RF_preds'], CELL_TYPES)
        plt.savefig(save_path.joinpath(save_fname+'-cfm_'+set_name+'.png'))
        
    joblib.dump(rf, save_path.joinpath(save_fname+'.joblib'))     
    
        