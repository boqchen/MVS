import numpy as np
import pandas as pd
import os
import glob 
from pathlib import Path
import json
import argparse
from scipy.stats import describe
from scipy.stats.stats import pearsonr
import torchvision.transforms.functional as ttf
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

import torch
import torch.nn as nn
from PIL import Image
import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure, StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.kid import KernelInceptionDistance
import sys 
root_code = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.insert(0, root_code)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from codebase.utils.constants import *
from codebase.utils.raw_utils import str2bool
from codebase.utils.eval_utils import *
from codebase.utils.metrics import *
from codebase.experiments.cgan3.loaders import *


parser = argparse.ArgumentParser(description='Evaluation on protein level extended: to calculate FID, KID, ')
parser.add_argument('--submission_id', type=str, required=True, default=None, help='Job submission_id')
parser.add_argument('--epoch', type=str, required=False, default='last', help='if last then takes the last one found, if best then searching for chkpt_selection info, or say "45K" or step size for which inference already done')
parser.add_argument('--level', type=int, required=False, default=2, help='Which resolution to use {2,4,6}')
parser.add_argument('--dataset', type=str, required=False, default="test", help='Which set from split to use {test, valid, train}')
parser.add_argument('--avg_kernel', type=int, required=False, default=32, help='Size of the averaging kernel (if 0, no averaging is applied)')
parser.add_argument('--project_path', type=str, required=False, default='/raid/sonali/project_mvs/', help='Path where all data, results etc for project reside')
parser.add_argument('--which_cluster', type=str, required=True, help='biomed or dgx:gpu')
parser.add_argument('--save_fname', type=str, required=False, default='eval_agg.csv', help='eval.csv when done per roi per marker, eval_agg.csv when done per oi but aggregated across markers')

# TODO: need to re-run without aggregation over markers (need to update dataloader)

args = parser.parse_args()
submission_id = args.submission_id
results_path = Path(args.project_path).joinpath('results')

# getting IMC true path 
job_args = json.load(open(Path(results_path).joinpath(submission_id, 'args.txt')))
protein_subset = get_protein_list(job_args['protein_set'])
channel_list = [protein2index[protein] for protein in protein_subset]
imc_roi_storage = get_imc_roi_storage(Path(args.project_path).joinpath('data', 'tupro'),
        job_args['imc_prep_seq'], str2bool(job_args['standardize_imc']), str2bool(job_args['scale01_imc']), job_args['cv_split'])
print(imc_roi_storage, channel_list, protein_subset)

# setting device 
dev0 = 'cuda:0' if 'dgx' not in args.which_cluster else 'cuda:' + args.which_cluster.split(':')[1]
dev0 = torch.device(dev0 if torch.cuda.is_available() else 'cpu')
print('Device: ', dev0)

# dataloader
batch_size = 1
which_step = '405K' # iterate through all and find the best one
imc_pred_paths = glob.glob(os.path.join(results_path, submission_id, args.dataset + '_images', 'step_' + args.epoch, 'level_' + str(args.level)) + '/*npy')
print(imc_pred_paths)

eval_ds = Eval_dataset(imc_pred_paths, imc_roi_storage, channel_list)
evalloader = DataLoader(eval_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=8, 
                            drop_last=True)

# Metrics initialisation
fid = FrechetInceptionDistance(feature=64, normalize=True).to(dev0)
kid = KernelInceptionDistance(subset_size=10, normalize=True).to(dev0)
ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, gaussian_kernel=True, kernel_size=25, sigma=1).to(dev0)

# loop though the data 
sample_roi_list = []
ms_ssim_list = []
fid_list = []
kid_list = []
# protein_list = []

# TODO: atm channels converted to batch and metrics eg FID give avg over batch, not for each protein -- re-run for best afterwards 
for i, batch in enumerate(evalloader):
    sample_roi = batch['sample_roi']
    print(sample_roi)

    # print(batch['imc_gt_path'])
    # print(batch['imc_pred_path'])    
    # print(batch['imc_gt'].shape, batch['imc_pred'].shape)

    imc_pred = batch['imc_pred'].squeeze(0).to(dev0)
    imc_gt = batch['imc_gt'].squeeze(0).to(dev0)

    # print('imc_pred: ', imc_pred.shape)

    # fid score
    fid.update(imc_gt, real=True)
    fid.update(imc_pred, real=False)
    fid_score = fid.compute()
    # print('fid_score: ', fid_score)

    # kid score
    kid.update(imc_gt, real=True)
    kid.update(imc_pred, real=False)
    kid_score = kid.compute()
    # print('kid_score: ', kid_score)

    # ms-ssim 
    ms_ssim_score = ms_ssim(imc_pred, imc_gt)
    # print('ms_ssim_score: ', ms_ssim_score)

    # append 
    sample_roi_list.append(sample_roi[0])
    fid_list.append(round(fid_score.item(), 4))
    kid_list.append(round(kid_score[0].item(), 4))
    ms_ssim_list.append(round(ms_ssim_score.item(), 4))

# metrics to pandas df 
df_fid = pd.DataFrame({'sample_roi': sample_roi_list, 'fid': fid_list})
df_kid = pd.DataFrame({'sample_roi': sample_roi_list, 'kid': kid_list})
df_msssim = pd.DataFrame({'sample_roi': sample_roi_list, 'msssim': ms_ssim_list})

# df_mean = df_msssim.msssim.agg(['mean', 'std'])
# print(df_mean)

# save eval metrics
SAVE_PATH = Path(results_path).joinpath(submission_id, args.dataset+'_eval', 'step_' + args.epoch, 'level_'+str(args.level), 'avgkernel_'+str(args.avg_kernel))
if not os.path.exists(SAVE_PATH):
    Path(SAVE_PATH).mkdir(parents=True)

df_fid.to_csv(SAVE_PATH.joinpath('fid-'+args.save_fname)) 
df_kid.to_csv(SAVE_PATH.joinpath('kid-'+args.save_fname)) 
df_msssim.to_csv(SAVE_PATH.joinpath('msssim-'+args.save_fname)) 

# df_msssim1 = pd.read_csv(SAVE_PATH.joinpath('msssim-'+args.save_fname))    
# df_mean1 = df_msssim1.msssim.agg(['mean', 'std'])
# print(df_mean1)
# print(SAVE_PATH.joinpath('fid-'+args.save_fname))
# print(SAVE_PATH.joinpath('kid-'+args.save_fname))
# print(SAVE_PATH.joinpath('msssim-'+args.save_fname))