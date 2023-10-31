import numpy as np
import pandas as pd
import os
from pathlib import Path
import json
from scipy.stats import describe
from scipy.stats.stats import pearsonr
import torchvision.transforms.functional as ttf
import torchvision.transforms as tt
import torch
import torch.nn as nn
from PIL import Image
import argparse
import warnings
warnings.filterwarnings("ignore")

from codebase.utils.constants import *
from codebase.utils.raw_utils import str2bool
from codebase.utils.eval_utils import * #get_protein_list, get_thrs_grid, binarize_array, img_preproc
from codebase.utils.metrics import *
#from codebase.experiments.cgan3.training_helpers import resize_tensor

parser = argparse.ArgumentParser(description='Evaluation on protein level.')
parser.add_argument('--submission_id', type=str, required=True, default=None, help='Job submission_id')
parser.add_argument('--epoch', type=str, required=False, default='last', help='if last then takes the last one found, if best then searching for chkpt_selection info, or say "45K" or step size for which inference already done')
parser.add_argument('--level', type=int, required=False, default=2, help='Which resolution to use {2,4,6}')
parser.add_argument('--data_set', type=str, required=False, default="test", help='Which set from split to use {test, valid, train}')
parser.add_argument('--sel_roi_name', type=str, required=False, default=None, help='Selected ROI to perform eval on; if not specified, then eval on all ROIs for which predicted images exist')
parser.add_argument('--project_path', type=str, required=False, default='/raid/sonali/project_mvs/', help='Path where all data, results etc for project reside')
parser.add_argument('--eval_metrics', type=str, required=False, default="pcorr", help='Which eval metrics to use, separated by comma {stats,pcorr,ssim,cwssim,dice,overlap,density_corr,perc_pos,hmi,psnr}')
parser.add_argument('--blur_sigma', type=int, required=False, default=0, help='Stdev the blurring Gaussian kernel (if 0, no blurring is applied)')
parser.add_argument('--kernel_width', type=int, required=False, default=3, help='Width of the Gaussian kernel used for blurring')
parser.add_argument('--avg_kernel', type=int, required=False, default=32, help='Size of the averaging kernel (if 0, no averaging is applied)')
parser.add_argument('--avg_stride', type=int, required=False, default=1, help='Stride for the averaging kernel (only used if avg_kernel>0)')
parser.add_argument('--ssim_kernel_sigma', type=int, required=False, default=1, help='Sigma of the Gaussian kernel used for SSIM')
parser.add_argument('--ssim_kernel_width', type=int, required=False, default=75, help='Width of the Gaussian kernel used for SSIM')
parser.add_argument('--cwssim_conv_width', type=int, required=False, default=30, help='Width of the convolution used for CW-SSIM')
parser.add_argument('--thrs_start', type=float, required=False, default=0, help='Starting value for binarization threshold grid')
parser.add_argument('--thrs_end', type=float, required=False, default=1, help='End value for binarization threshold grid')
parser.add_argument('--thrs_step', type=float, required=False, default=0.1, help='Step value for binarization threshold grid')
parser.add_argument('--thrs_how', type=str, required=False, default="quantile", help='Method for setting binarization threshold grid {step, quantile, cohort}')
parser.add_argument('--thrs_cohort_path', type=str, required=False, default="data/tupro/imc_updated/agg_masked_data-raw_clip99_arc_otsu3_std_minmax_split3-r5-train_quantiles.tsv", help='Path to cohort (train set) quantiles; only used if thrs_how==cohort')
parser.add_argument('--densitycorr_metric', type=str, required=False, default="pcorr", help='Which correlation coefficient to use to compare densities {pcorr, spcorr}')
parser.add_argument('--densitycorr_px', type=int, required=False, default=32, help='Desired resolution in px to compute density; n_bins=1000//densitycorr_px')
parser.add_argument('--add_tumorprots', type=str2bool, required=False, default=False, help='Whether to add "tumor_prots" column, by taking max across tumor proteins')
parser.add_argument('--which_cluster', type=str, required=True, help='biomed or dgx:gpu')

args = parser.parse_args()

job_name = args.submission_id
eval_metrics = args.eval_metrics.split(',')
epoch = args.epoch
if epoch == 'best':
    step_name = get_best_epoch_w_imgs(args.project_path, job_name, level=args.level, data_set='valid')
elif epoch == 'last':
    step_name = get_last_epoch_w_imgs(args.project_path, job_name, level=args.level, data_set=args.data_set)
else:
    step_name = str(epoch)
print('At', step_name)

# overwritten from constants due to use of both biomed and dgx
RESULTS_DIR = Path(args.project_path).joinpath('results')
INPUT_PATH = Path(RESULTS_DIR).joinpath(job_name, args.data_set+'_images',  'step_' + step_name, 'level_'+str(args.level))

print(INPUT_PATH)
s_rois = [x.split('.')[0] for x in os.listdir(INPUT_PATH)] if args.sel_roi_name is None else [args.sel_roi_name]
print(s_rois)

# Get info form job arguments
job_args = json.load(open(Path(RESULTS_DIR).joinpath(job_name, 'args.txt')))
protein_subset = get_protein_list(job_args['protein_set'])
channel_list = [protein2index[protein] for protein in protein_subset]
IMC_ROI_STORAGE = get_imc_roi_storage(Path(args.project_path).joinpath('data', 'tupro'),
        job_args['imc_prep_seq'], str2bool(job_args['standardize_imc']), str2bool(job_args['scale01_imc']), job_args['cv_split'])
if args.add_tumorprots:
    protein_subset.append('tumor_prots')

# setting device 
dev0 = 'cuda:0' if 'dgx' not in args.which_cluster else 'cuda:' + args.which_cluster.split(':')[1]
dev0 = torch.device(dev0 if torch.cuda.is_available() else 'cpu')
print('Device: ', dev0)

corrs = dict()
pvals = dict()
ssim_df_all = pd.DataFrame()
binary_eval_df_all = pd.DataFrame()
hmi_df_all = pd.DataFrame()
psnr_df_all = pd.DataFrame()
stats_df_all = pd.DataFrame()
for s_roi in s_rois:
    print(s_roi)
    # ----- Loading and preprocessing images -----
    # Load GT IMC (and subset GT IMC to channels in pred IMC)
    imc_gt = np.load(os.path.join(IMC_ROI_STORAGE, s_roi+'.npy'), mmap_mode='r')[:,:,channel_list]
    # No Gaussian blur on GT (thus blur_sigma=0); downsmapling to match pred_IMC downsampling (if present)
    imc_gt = preprocess_img(imc_gt, dev0, downsample_factor=1000//(4000//2**args.level), kernel_width=args.kernel_width, blur_sigma=0, avg_kernel=args.avg_kernel, avg_stride=args.avg_stride)
    
    # Load pred IMC (note: channels are already in the right order)
    imc_pred = np.load(INPUT_PATH.joinpath(s_roi+'.npy'), mmap_mode='r')
    imc_pred = preprocess_img(imc_pred, dev0, downsample_factor=1, kernel_width=args.kernel_width, blur_sigma=args.blur_sigma, avg_kernel=args.avg_kernel, avg_stride=args.avg_stride)
   
    # Make sure GT and pred IMC images have the same shapes and number of channels
    assert np.isclose(imc_gt.shape[1],imc_pred.shape[1]) and np.isclose(imc_gt.shape[2],imc_pred.shape[2]), 'GT and pred IMC shapes do not match!'
    assert np.isclose(imc_gt.shape[0],imc_pred.shape[0]), 'Number of channels in GT and pred IMC do not match!'

    if args.add_tumorprots:
        imc_gt_tumor = get_tumor_prots_signal(imc_gt, protein_subset)
        imc_gt = np.append(imc_gt, imc_gt_tumor, axis=2)
        imc_pred_tumor = get_tumor_prots_signal(imc_pred, protein_subset)
        imc_pred = np.append(imc_pred, imc_pred_tumor, axis=2)
        
    ### calculate pixel-wise Pearson's correlation coefficient
    if ('pcorr' in eval_metrics) or ('stats' in eval_metrics):
        corrs[s_roi] = dict()
        pvals[s_roi] = dict()
        for i,protein in enumerate(protein_subset):
            print(i, protein)
            imc_gt_flatten = imc_gt[:,:,i].flatten()
            imc_pred_flatten = imc_pred[:,:,i].flatten()
            if 'pcorr' in eval_metrics:
                corrs[s_roi][protein], pvals[s_roi][protein] = pearsonr(imc_gt_flatten, imc_pred_flatten)
            if 'stats' in eval_metrics:
                stats_df = describe(imc_gt_flatten)
                stats_df = pd.DataFrame([stats_df], columns=[x+'_gt' for x in stats_df._fields], index=[protein])
                stats_df_pred = describe(imc_pred_flatten)
                stats_df_pred = pd.DataFrame([stats_df_pred], columns=[x+'_pred' for x in stats_df_pred._fields], index=[protein])  
                stats_df = pd.concat([stats_df, stats_df_pred], axis=1)             
                stats_df.loc[:,['sample_id','roi']] = s_roi.split('_')
                stats_df_all = pd.concat([stats_df_all, stats_df])
    
    ### calculate SSIM and/or CW-SSIM scores and/or peak signal-to-noise ratio (PSNR)
    if ('ssim' in eval_metrics) or ('cwssim' in eval_metrics) or ('psnr' in eval_metrics):
        # TODO: best would be to do it as the last eval and modify the img directly other than creating a copy
        ssim_df = pd.DataFrame(index=protein_subset, columns=[x for x in ['ssim','cwssim'] if x in eval_metrics])
        for i,protein in enumerate(protein_subset):
            imc_gt_ssim = prep_im_for_ssim(imc_gt[:,:,i], scale_01=True)
            imc_pred_ssim = prep_im_for_ssim(imc_pred[:,:,i], scale_01=True)
            if 'ssim' in eval_metrics:
                ssim_val = get_ssim(imc_gt_ssim, imc_pred_ssim, gaussian_kernel_sigma=args.ssim_kernel_sigma, gaussian_kernel_width=args.ssim_kernel_width)
                ssim_df.loc[protein, 'ssim'] = ssim_val
            if 'cwssim' in eval_metrics:
                cwssim_val = get_cwssim(imc_gt_ssim, imc_pred_ssim, conv_width=args.cwssim_conv_width)
                ssim_df.loc[protein, 'cwssim'] = cwssim_val
            if 'psnr' in eval_metrics:
                psnr_val = get_psnr(imc_gt_ssim, imc_pred_ssim)
                ssim_df.loc[protein, 'psnr'] = psnr_val
        ssim_df['sample_id'] = s_roi.split('_')[0]
        ssim_df['roi'] = s_roi.split('_')[1]
        ssim_df_all = pd.concat([ssim_df_all, ssim_df])

    ### calculate score based on binarized arrays (dice, overlap and perc positive pixels)
    if len([x for x in ['dice', 'overlap', 'perc_pos'] if x in eval_metrics])>0:
        if args.thrs_how=='cohort':
            cohort_quantiles = pd.read_csv(Path(args.project_path).joinpath(args.thrs_cohort_path), sep='\t', index_col=[0])
        for i,protein in enumerate(protein_subset):
            # get evals using threshold-based binarization
            thrs_cohort_df = None
            if args.thrs_how=='cohort':
                thrs_cohort_df = cohort_quantiles.loc[:,[protein]]
            binary_eval_df = get_thrs_metrics_protein(imc_gt[:,:,i], imc_pred[:,:,i], dice_score=('dice' in eval_metrics), overlap=('overlap' in eval_metrics), perc_pos=('perc_pos' in eval_metrics), thrs_start=args.thrs_start, thrs_end=args.thrs_end, thrs_step=args.thrs_step, thrs_how=args.thrs_how, thrs_cohort_df=thrs_cohort_df)
            binary_eval_df.loc[:,['sample_id','roi']] = s_roi.split('_')
            binary_eval_df.index = [protein]
            binary_eval_df_all = pd.concat([binary_eval_df_all, binary_eval_df])

    ### calculate histogram based mutual information
    if ('hmi' in eval_metrics):
        hmi_df = pd.DataFrame(index=protein_subset)
        for i,protein in enumerate(protein_subset):
            hmi_df.loc[protein, 'hmi'] = histogram_mutual_information(imc_gt[:,:,i], imc_pred[:,:,i])
        hmi_df.loc[:,['sample_id','roi']] = s_roi.split('_')
        hmi_df_all = pd.concat([hmi_df_all, hmi_df])
    
if 'pcorr' in eval_metrics:        
    corrs_df = pd.DataFrame().from_dict(corrs)
    corrs_df.index.name = 'protein'
    corrs_df = corrs_df.reset_index().melt(id_vars='protein', var_name='sample_roi', value_name='pcorr')
    pvals_df = pd.DataFrame().from_dict(pvals)
    pvals_df.index.name = 'protein'
    pvals_df = pvals_df.reset_index().melt(id_vars='protein', var_name='sample_roi', value_name='pval')
    corrs_df = corrs_df.merge(pvals_df, on=['sample_roi', 'protein'], how='left')
    corrs_df['sample_id'] = [x.split('_')[0] for x in corrs_df['sample_roi']]
    corrs_df['roi'] = [x.split('_')[1] for x in corrs_df['sample_roi']]

### save results
SAVE_PATH = Path(RESULTS_DIR).joinpath(job_name, args.data_set+'_eval', 'step_' + step_name, 'level_'+str(args.level), 'avgkernel_'+str(args.avg_kernel))
save_fname = 'eval.csv' if args.sel_roi_name is None else 'eval-'+args.sel_roi_name+'.csv'
if not os.path.exists(SAVE_PATH):
    Path(SAVE_PATH).mkdir(parents=True)
if 'pcorr' in eval_metrics:
    corrs_df.to_csv(SAVE_PATH.joinpath('pcorr-'+save_fname)) 
for eval_metric in ['ssim', 'cwssim', 'psnr']:   
    if (eval_metric in eval_metrics):
        ssim_df_all.loc[:,[eval_metric, 'sample_id','roi']].to_csv(SAVE_PATH.joinpath(eval_metric+'-'+save_fname))
if len([x for x in eval_metrics if x in ['dice', 'overlap' ,'density_corr', 'perc_pos']])>0:
    binary_eval_df_all.to_csv(SAVE_PATH.joinpath('binary-'+save_fname))     
if ('hmi' in eval_metrics):
    hmi_df_all.to_csv(SAVE_PATH.joinpath('hmi-'+save_fname))  
if ('stats' in eval_metrics):
    stats_df_all.loc[:,['min_gt', 'max_gt']] = [print(x,y) for x,y in stats_df_all['minmax_gt']]
    stats_df_all.loc[:,['min_pred', 'max_pred']] = [print(x,y) for x,y in stats_df_all['minmax_pred']]
    stats_df_all.to_csv(SAVE_PATH.joinpath('stats-'+save_fname))  
# save argument values into a txt file
with open(Path(SAVE_PATH).joinpath('eval_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
