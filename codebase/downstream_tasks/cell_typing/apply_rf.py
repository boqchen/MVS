import openslide
import os
import numpy as np
import random
import json
import joblib
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad
from sklearn.metrics import confusion_matrix as cfm
import argparse
from pathlib import Path

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as ttf


from codebase.utils.dataset_utils import *
from codebase.utils.constants import *
from codebase.utils.cv_split import kfold_splits,get_patient_samples
from codebase.utils.eval_utils import *
from codebase.downstream_tasks.cell_typing.random_forest import *

## TODO: add option for a sliding window/blur predictions
def blur_smooth_array(array, blur_sigma, avg_kernel):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    array = array.transpose((2, 0, 1))
    array = np.ascontiguousarray(array)
    array = torch.from_numpy(array).float().unsqueeze(0)
    array = array.to(device)

    if blur_sigma > 0:
        spatial_denoise = torchvision.transforms.GaussianBlur(3, sigma=blur_sigma)
        array = spatial_denoise(array)
    if avg_kernel>0:
        avg_pool = nn.AvgPool2d(kernel_size=avg_kernel, padding=int(np.floor(avg_kernel/2)), stride=1)
        array = avg_pool(array)
    return array.detach().cpu().numpy()[0].transpose((1, 2, 0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RF clasifier using train set.')
    parser.add_argument('--save_path', type=str, required=False, default='/cluster/work/grlab/projects/projects2021-multivstain/results/', help='Path to save the predictions')
    parser.add_argument('--model_path', type=str, required=False, default='/cluster/work/grlab/projects/projects2021-multivstain/results', help='Path with model logs')
    parser.add_argument('--set', type=str, required=False, default="test", help='Which set from split to use {test, valid, train}')
    parser.add_argument('--model_type', type=str, required=False, default="cgan2", help='Which model to use')
    parser.add_argument('--submission_id', type=str, required=True, default=None, help='Job submission id')
    parser.add_argument('--n_estimators', type=int, required=False, default=75, help='Number of trees to use')
    parser.add_argument('--ct_level', type=str, required=False, default='CT2', help='Cell-typing level to use for training RF')
    parser.add_argument('--epoch', type=int, required=False, default=None, help='Which epoch to use (number), if not specified the last computed is used')
    parser.add_argument('--suppress_q', type=float, required=False, default=0.1, help='Quantile below which the signal is considered noise and is set to 0')
    parser.add_argument('--blur_sigma', type=int, required=False, default=0, help='Stdev the blurring Gaussian kernel (if 0, no blurring is applied)')
    parser.add_argument('--avg_kernel', type=int, required=False, default=0, help='Size of the averaging kernel (if 0, no averaging is applied)')
    
    args = parser.parse_args()
    model_path = Path(args.model_path).joinpath(args.submission_id)
    
    save_path = Path(args.save_path).joinpath(args.submission_id, args.set+'_ct','epoch'+str(args.epoch))
    if not os.path.exists(save_path):
        save_path.mkdir(parents=True, exist_ok=False)

    # Define subset of predicted proteins and cv_split (based on args.txt):
    job_args = pd.read_json(Path(args.model_path).joinpath(args.submission_id,'args.txt'), orient='index')
    protein_set_name = job_args.loc['protein_set',:].values[0]
    protein_set = get_protein_list(protein_set_name)
    cv_split = job_args.loc['cv_split',:].values[0]
    
    # Load trained RF
    rf_fname = '-'.join(['rf', args.ct_level, protein_set_name, 'ntrees'+str(args.n_estimators)])+'.joblib'
    print(Path(META_DIR).joinpath(cv_split, rf_fname))
    assert os.path.exists(Path(META_DIR).joinpath('rf',cv_split, rf_fname)), 'RF for selected settings was not trained'
    rf = joblib.load(Path(META_DIR).joinpath('rf',cv_split, rf_fname))
    
    # Load image with predictions
    img_path = model_path.joinpath(args.set+'_images','epoch'+str(args.epoch))
    for roi in os.listdir(img_path):
        print(roi)
        df_roi_pred = np.load(img_path.joinpath(roi))
        for i,prot_name in enumerate(protein_set):
            prot_idx = protein2index[prot_name]
            df_roi_pred[:,:,i] = destandardize_img(df_roi_pred[:,:,i], EXPRS_AVG[prot_idx], EXPRS_STDEV[prot_idx])
            if args.suppress_q > 0:
                df_roi_pred[:,:,i] = np.apply_along_axis(suppress_to_zero,1,df_roi_pred[:,:,i], q=args.suppress_q)
        df_roi_pred = blur_smooth_array(df_roi_pred, args.blur_sigma, args.avg_kernel)
        df_roi_pred = df_roi_pred.reshape((-1, len(protein_set)))
        
        df_roi_gt = np.load(BINARY_IMC_ROI_STORAGE+roi)
        df_roi_gt = blur_smooth_array(df_roi_gt, args.blur_sigma, args.avg_kernel)
        org_shape = df_roi_gt.shape
        prot_idx = [protein2index[prot_name] for prot_name in protein_set]
        df_roi_gt = df_roi_gt[:,:,prot_idx]
        if args.suppress_q > 0:
            for i in range(df_roi_gt.shape[2]):
                df_roi_pred[:,:,i] = np.apply_along_axis(suppress_to_zero,1,df_roi_gt[:,:,i], q=args.suppress_q)
        df_roi_gt = df_roi_gt.reshape((-1, len(protein_set)))
        
        preds = get_manual_aggregation(rf, df_roi_pred, CELL_TYPES)
        preds_gt = get_manual_aggregation(rf, df_roi_gt, CELL_TYPES)
        save_fname = '-'.join(['rf_pred', args.ct_level, protein_set_name, 'ntrees'+str(args.n_estimators), 'suppress'+str(args.suppress_q).replace('.','_'), 'blur'+str(args.blur_sigma),'kernel'+str(args.avg_kernel),'long',roi])
        np.save(save_path.joinpath(save_fname), preds)
        np.save(save_path.joinpath(save_fname.replace('rf_pred','rf_gt_pred')), preds_gt)
        
        preds = preds.reshape(org_shape[0], org_shape[1], len(CELL_TYPES))
        preds_gt = preds_gt.reshape(org_shape[0], org_shape[1], len(CELL_TYPES))
        save_fname = '-'.join(['rf_pred', args.ct_level, protein_set_name, 'ntrees'+str(args.n_estimators),'suppress'+str(args.suppress_q).replace('.','_'),'blur'+str(args.blur_sigma),'kernel'+str(args.avg_kernel), 'wide',roi])
        np.save(save_path.joinpath(save_fname), preds)
        np.save(save_path.joinpath(save_fname.replace('rf_pred','rf_gt_pred')), preds_gt)

        
        
