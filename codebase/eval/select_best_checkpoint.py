import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from scipy.stats.stats import pearsonr
import json
import os
import warnings

import torch
import torchvision
import torch.nn as nn

from codebase.utils.constants import *
from codebase.utils.inference_utils import *
from codebase.utils.eval_utils import *# get_protein_list, get_ordered_epoch_names
from codebase.experiments.cgan3.training_helpers import resize_tensor, str2bool

parser = argparse.ArgumentParser(description='Checkpoint selection.')
parser.add_argument('--project_path', type=str, required=False, default='/raid/sonali/project_mvs/', help='Path where all data, results etc for project reside')
parser.add_argument('--submission_id', type=str, required=True, default=None, help='')
parser.add_argument('--start_epoch', type=int, required=False, default=1, help='Compute eval for checkpoint selection starting at start epoch, eg default starting 1st saved checkpoint')
parser.add_argument('--every_x_epoch', type=int, required=False, default=1, help='Compute eval for checkpoint selection every x epoch')
parser.add_argument('--level', type=int, required=False, default=2, help='Which resolution to use {2,4,6}')
parser.add_argument('--avg_kernel', type=int, required=False, default=64, help='Size of the averaging kernel (if 0, no averaging is applied)')
parser.add_argument('--avg_stride', type=int, required=False, default=1, help='Stride for the averaging kernel (only used if avg_kernel>0)')
parser.add_argument('--data_set', type=str, required=False, default="valid", help='Which set from split to use {test, valid, train}')
parser.add_argument('--which_cluster', type=str, required=True, help='biomed or dgx:gpu')


args = parser.parse_args()
start_epoch = args.start_epoch
every_x_epoch = args.every_x_epoch

# load job arguments / settings
job_args = json.load(open(Path(args.project_path).joinpath('results',args.submission_id,'args.txt')))
# define paths
IMC_ROI_STORAGE = get_imc_roi_storage(Path(args.project_path).joinpath('data', 'tupro'), job_args['imc_prep_seq'], str2bool(job_args['standardize_imc']), str2bool(job_args['scale01_imc']), job_args['cv_split'])
MODEL_PATH = Path(args.project_path).joinpath('results', args.submission_id, 'tb_logs')
CV_SPLIT_ROIS_PATH = Path(args.project_path).joinpath(CV_SPLIT_ROIS_PATH)
BINARY_HE_ROI_STORAGE = get_he_roi_storage(Path(args.project_path).joinpath('data/tupro'), job_args['which_HE'])
# get sample_rois from validation in a given cv split
cv_split = job_args['cv_split']
cv = json.load(open(CV_SPLIT_ROIS_PATH))
sample_rois = cv[cv_split][args.data_set]
# define protein list (based on args.txt)
protein_list = get_protein_list(job_args['protein_set'])
channel_list = [protein2index[protein] for protein in protein_list]
model_depth = job_args['model_depth']

# set device
dev0 = 'cuda:0' if 'dgx' not in args.which_cluster else 'cuda:' + args.which_cluster.split(':')[1]
dev0 = torch.device(dev0 if torch.cuda.is_available() else 'cpu')
print('Device: ', dev0)

# steps for which we have checkpoints saved eg 5K, 10K etc 
step_names = get_ordered_epoch_names(MODEL_PATH)
print(step_names)

# ----- check if data from previous runs of checkpoint selection are available, otherwise create folder ----- 
SAVE_PATH = Path(args.project_path).joinpath('results', args.submission_id, 'chkpt_selection')
save_fname = 'level_'+str(args.level)+'-'+args.data_set
valid_evals = pd.DataFrame()

fname = []
if os.path.exists(SAVE_PATH): 
    fname = [x for x in os.listdir(SAVE_PATH) if save_fname in x]
print('fname: ', fname)
# if not os.path.exists(SAVE_PATH):
if len(fname)==0:
    Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
    step_names = step_names[start_epoch-1::every_x_epoch]
    print(step_names)
else:
    valid_evals = pd.read_csv(SAVE_PATH.joinpath(fname[0]), index_col=[0])
    print(valid_evals)
    # find last epoch for which eval was computed
    print(valid_evals['step_name'])  

    start_step = valid_evals['step_name'].iloc[-1]
    print('start_step: ', start_step)
    start_epoch =  step_names.index(start_step) + 1             
    print('start_epoch: ', start_epoch)
    step_names = step_names[start_epoch::every_x_epoch]
    print(step_names)

    if every_x_epoch > 1:
        start_epoch = start_epoch + every_x_epoch

INFERENCE_PATH = Path(args.project_path).joinpath('results', args.submission_id, 'valid_images')
corrs_all = dict()

for step_name in step_names:
    print(step_name)

    # ----- check of predictions already available, otherwise perform inference -----
    # find last pth with this epoch number
    pth_name = step_name + '_translator.pth'
    print('pth_name: ', pth_name)  
    if os.path.exists(INFERENCE_PATH.joinpath('step_'+step_name)):
        pred_avail = True
    else:
        pred_avail = False
        network = torch.load(MODEL_PATH.joinpath(pth_name), map_location=torch.device(dev0))
        network.to(dev0)
        network.eval()
        
    corrs = dict()
    for s_roi in sample_rois:
        print(s_roi)
        # load or predict IMC (if prediction not available)
        if pred_avail:
            imc_pred = np.load(INFERENCE_PATH.joinpath('step_'+step_name, 'level_'+str(args.level), s_roi + ".npy"), mmap_mode='r')
        else:
            he_roi_np_lvl0 = np.load(BINARY_HE_ROI_STORAGE.joinpath(s_roi + ".npy"), mmap_mode='r')
            he_roi_tensor_lvl0 = get_tensor_from_numpy(he_roi_np_lvl0) 
            # get imc desired shapes based on input HE image
            imc_desired_shapes = get_target_shapes(model_depth, he_roi_np_lvl0.shape[0])
            # pad image to make compatible with model (eg /2)
            he_roi_tensor_lvl0 = pad_img(he_roi_tensor_lvl0, he_roi_np_lvl0.shape[0]).to(dev0)
            # predict IMC 
            with torch.no_grad():
                pred_imc_roi_tensor = network(he_roi_tensor_lvl0)
            imc_pred = pred_imc_roi_tensor[-(args.level//2)]
            imc_pred = torchvision.transforms.CenterCrop([imc_desired_shapes[(args.level//2) - 1], imc_desired_shapes[(args.level//2) - 1]])(imc_pred)
            
        # averaging kernel to counteract slice-slice discrepancy
        if args.avg_kernel>0:
            if torch.is_tensor(imc_pred)==False:
                imc_pred = get_tensor_from_numpy(imc_pred)
            avg_pool = nn.AvgPool2d(kernel_size=args.avg_kernel, padding=int(np.floor(args.avg_kernel/2)), stride=args.avg_stride)
            imc_pred = avg_pool(imc_pred)
            imc_pred = imc_pred[0].detach().cpu().numpy().transpose((1, 2, 0))

        if torch.is_tensor(imc_pred)==True:    
            imc_pred[0].detach().cpu().numpy().transpose((1, 2, 0))
            
        # load GT IMC
        imc_np_gt = np.load(os.path.join(IMC_ROI_STORAGE, s_roi+'.npy'), mmap_mode='r')
        # subset GT IMC to channels in pred IMC
        imc_np_gt = imc_np_gt[:,:,channel_list]
        imc_gt = torch.from_numpy(imc_np_gt.transpose(2,0,1))
        # downsample GT if pred IMC downsampled
        if args.level > 2:
            downsample_factor = 1000//(4000//2**args.level)
            imc_gt = resize_tensor(imc_gt, imc_np_gt.shape[0]//downsample_factor)[0]
        #imc_gt.to(dev0)
        if args.avg_kernel>0:
            imc_gt = avg_pool(imc_gt)
        imc_gt = imc_gt.detach().cpu().numpy().transpose((1, 2, 0))
        
        # make sure GT and pred IMC images have the same shapes and number of channels
        assert np.isclose(imc_gt.shape[1],imc_pred.shape[1]) and np.isclose(imc_gt.shape[2],imc_pred.shape[2]), 'GT and pred IMC shapes do not match!'
        assert np.isclose(imc_gt.shape[0],imc_pred.shape[0]), 'Number of channels in GT and pred IMC do not match!'
        
        # ----- Calculate Pearson's correlation for each protein -----
        corrs[s_roi] = dict()
        with warnings.catch_warnings():
            # to suppress PearsonRConstantInputWarning
            warnings.simplefilter("ignore")
            for i,protein in enumerate(protein_list):
                corrs[s_roi][protein], _ = pearsonr(imc_gt[:,:,i].flatten(), imc_pred[:,:,i].flatten())
            
        corrs_df = pd.DataFrame().from_dict(corrs)
        corrs_df.index.name = 'protein'
        corrs_df = corrs_df.reset_index().melt(id_vars='protein', var_name='sample_roi', value_name='pcorr')
    # TODO: implement other versions of aggregating
    corrs_all[step_name] = corrs_df.groupby('protein').pcorr.median()
    
corrs_df = pd.DataFrame().from_dict(corrs_all)
corrs_df.index.name = 'protein'
corrs_df = corrs_df.reset_index().melt(id_vars='protein', var_name='step_name', value_name='pcorr')
corrs_df_agg = corrs_df.groupby('step_name').pcorr.median().to_frame('agg_per_epoch').reset_index()
corrs_df = corrs_df.merge(corrs_df_agg, on='step_name', how='left')
valid_evals = pd.concat([valid_evals, corrs_df])
# save valid_evals
valid_evals.to_csv(SAVE_PATH.joinpath('pcorr_across_epochs-'+save_fname+'.csv'))

# ----- find and save the best epoch -----
best_step_info = dict()
best_step = valid_evals.drop_duplicates('step_name').loc[:,['step_name','agg_per_epoch']].sort_values('agg_per_epoch', ascending=True).step_name.values[-1]
best_step_info['best_step'] = best_step

best_step_info['chkpt_file'] = str(best_step) + '_translator.pth'
print(best_step_info)

with open(SAVE_PATH.joinpath('best_epoch-'+save_fname+'.txt'), 'w') as f:
    json.dump(best_step_info, f, indent=2)
# save job arguments
with open(SAVE_PATH.joinpath('selection_args-'+save_fname+'.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)




