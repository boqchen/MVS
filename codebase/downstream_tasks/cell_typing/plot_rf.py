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
from codebase.utils.viz_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RF clasifier using train set.')
    parser.add_argument('--save_path', type=str, required=False, default='/cluster/work/grlab/projects/projects2021-multivstain/results/', help='Path to save the predictions')
    parser.add_argument('--model_path', type=str, required=False, default='/cluster/work/grlab/projects/projects2021-multivstain/results', help='Path with model logs')
    parser.add_argument('--set', type=str, required=False, default="test", help='Which set from split to use {test, valid, train}')
    parser.add_argument('--model_type', type=str, required=False, default="cgan2", help='Which model to use')
    parser.add_argument('--submission_id', type=str, required=True, default=None, help='Job submission id')
    parser.add_argument('--n_estimators', type=int, required=False, default=75, help='Number of trees to use')
    parser.add_argument('--ct_level', type=str, required=False, default='CT2', help='Cell-typing level to use for training RF')
    parser.add_argument('--ntop_epochs', type=int, required=False, default=3, help='Number of top epochs to use')
    parser.add_argument('--epoch_selection', type=str, required=False, default='last', help='How to choose epochs {best, last}')
    parser.add_argument('--blur_sigma', type=int, required=False, default=0, help='Stdev the blurring Gaussian kernel (if 0, no blurring is applied)')
    parser.add_argument('--avg_kernel', type=int, required=False, default=0, help='Size of the averaging kernel (if 0, no averaging is applied)')
    
    args = parser.parse_args()
    model_path = Path(args.model_path).joinpath(args.submission_id)
    data_set = args.set
    ntop_epochs = args.ntop_epochs
    
    save_path = Path(args.save_path).joinpath(args.submission_id, args.set+'_plots_ct')
    if not os.path.exists(save_path):
        save_path.mkdir(parents=True, exist_ok=False)
    pred_img_path = model_path.joinpath('test_ct')

    # Define subset of predicted proteins and cv_split (based on args.txt):
    job_args = pd.read_json(Path(args.model_path).joinpath(args.submission_id,'args.txt'), orient='index')
    protein_set_name = job_args.loc['protein_set',:].values[0]
    protein_set = get_protein_list(protein_set_name)
    
    checkpoint_path = Path(str(pred_img_path).replace('test_ct','top_checkpoint'))
    if args.epoch_selection == 'best':
        if os.path.exists(checkpoint_path):
            top_epochs_eval = pd.read_csv(checkpoint_path.joinpath('top5_epochs-dice3-blur0.tsv'), sep='\t', header=None)
            top_epochs_eval = top_epochs_eval.iloc[:,0].to_list()[0:ntop_epochs]
        else:
            top_epochs_eval = ['epoch'+str(x) for x in np.sort([int(x.split('_')[-1].split('.')[0]) for x in os.listdir(model_path.joinpath('tb_logs')) if 'pth' in x and 'translator' in x])[-ntop_epochs:]]
    else:
        top_epochs_eval = ['epoch'+str(x) for x in np.sort([int(x.split('_')[-1].split('.')[0]) for x in os.listdir(model_path.joinpath('tb_logs')) if 'pth' in x and 'translator' in x])[-ntop_epochs:]]
        
    types = CELL_TYPES
    figsize = (14,14)
    
    settings = '-'.join([args.ct_level, protein_set_name, 'ntrees'+str(args.n_estimators),'blur'+str(args.blur_sigma),'kernel'+str(args.avg_kernel)])
    epochs = '_'.join(top_epochs_eval)
    save_fname = '-'.join(['rf_ct', settings, epochs])+'.png'
    pred_img_path = Path(pred_img_path) #model_path.joinpath(model_name, 'valid_images')
    files = [x for x in os.listdir(pred_img_path.joinpath(os.listdir(pred_img_path)[0])) if 'rf_pred' in x and 'wide' in x and settings in x]
    print('Found '+str(len(files))+' predictions')
    for fname in files:
        plot_ct_top(pred_img_path, fname, types, sel_types=types,top_epochs=top_epochs_eval, figsize=figsize, save_fname=save_path.joinpath(save_fname), plot_show=False)
        plt.close()
        
        
