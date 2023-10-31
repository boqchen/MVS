import numpy as np
import pandas as pd
import os
from scipy.stats.stats import pearsonr
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as ttf

from codebase.utils.constants import *
from codebase.utils.raw_utils import *
from codebase.utils.dataset_utils import *
from codebase.utils.cv_split import kfold_splits
from codebase.utils.eval_utils import *


parser = argparse.ArgumentParser(description='Evaluation on a selected set.')
parser.add_argument('--save_path', type=str, required=False, default='/cluster/work/grlab/projects/projects2021-multivstain/results', help='Path to save the predictions')
parser.add_argument('--model_path', type=str, required=False, default='/cluster/work/grlab/projects/projects2021-multivstain/results', help='Path with model logs')
parser.add_argument('--set', type=str, required=False, default="test", help='Which set from split to use {test, valid, train}')
parser.add_argument('--model_type', type=str, required=False, default="cgan2", help='Which model to use')
parser.add_argument('--submission_id', type=str, required=True, default=None, help='Job submission id')
parser.add_argument('--epoch', type=int, required=True, default=None, help='Which epoch to use (number)')
parser.add_argument('--blur_sigma', type=int, required=False, default=1, help='Stdev the blurring Gaussian kernel (if 0, no blurring is applied)')
parser.add_argument('--avg_kernel', type=int, required=False, default=75, help='Size of the averaging kernel (if 0, no averaging is applied)')

args = parser.parse_args()
save_path = Path(args.save_path)
data_set = args.set
epoch = args.epoch
model_name = args.submission_id
#model_path = Path(args.model_path).joinpath(model_name, 'tb_logs')

epoch = args.epoch
    
save_path_eval = save_path.joinpath(model_name, args.set+'_roi_eval')
if not os.path.exists(save_path_eval):
    save_path_eval.mkdir(parents=True, exist_ok=False)

# define subset of predicted proteins and cv_split (based on args.txt):
job_args = pd.read_json(Path(args.model_path).joinpath(model_name,'args.txt'), orient='index')
protein_set_name = job_args.loc['protein_set',:].values[0]
protein_set = get_protein_list(protein_set_name)
cv_split = job_args.loc['cv_split',:].values[0]
aligns = kfold_splits[cv_split][data_set]

if args.avg_kernel>0:
    avg_pool = nn.AvgPool2d(kernel_size=args.avg_kernel, padding=int(np.floor(args.avg_kernel/2)), stride=1)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dict_list = []    
for ar_idx, ar in enumerate(aligns):
    print(ar_idx)
    # load and prepare data
    #he_roi_np_lvl0 = np.load(BINARY_HE_ROI_STORAGE + ar["sample"] + '_' + ar["ROI"] + ".npy")
    imc_roi_np_lvl2 = np.load(BINARY_IMC_ROI_STORAGE + ar["sample"] + '_' + ar["ROI"] + ".npy")
    imc_roi_np_lvl2 = standardize_imc_array(imc_roi_np_lvl2)

#     he_roi_tensor_lvl0 = he_roi_np_lvl0.transpose((2, 0, 1))
#     he_roi_tensor_lvl0 = np.ascontiguousarray(he_roi_tensor_lvl0)
#     he_roi_tensor_lvl0 = torch.from_numpy(he_roi_tensor_lvl0).float().unsqueeze(0)
#     he_roi_tensor_lvl0 = he_roi_tensor_lvl0.to(device)

#     he_roi_tensor_lvl2 = ttf.resize(he_roi_tensor_lvl0, 1000)
#     he_roi_tensor_lvl4 = ttf.resize(he_roi_tensor_lvl0, 250)
#     he_roi_tensor_lvl6 = ttf.resize(he_roi_tensor_lvl0, 63)

#     he_roi_np_lvl2 = cv2.resize(he_roi_np_lvl0, dsize=(1000, 1000))

    imc_roi_tensor_lvl2_raw = imc_roi_np_lvl2.transpose((2, 0, 1))
    imc_roi_tensor_lvl2_raw = np.ascontiguousarray(imc_roi_tensor_lvl2_raw)
    imc_roi_tensor_lvl2_raw = torch.from_numpy(imc_roi_tensor_lvl2_raw).float().unsqueeze(0)
    imc_roi_tensor_lvl2_raw = imc_roi_tensor_lvl2_raw.to(device)
    
    # Load predicted images
    high_res_fake_imc = np.load(Path(args.model_path).joinpath(model_name,data_set+'_images', 'epoch'+str(args.epoch), ar["sample"] + '_' + ar["ROI"] + ".npy"))
    #import pdb; pdb.set_trace()
    if len(high_res_fake_imc.shape)<3:
        high_res_fake_imc = np.expand_dims(high_res_fake_imc, axis=-1)
        #high_res_fake_imc = torch.from_numpy(high_res_fake_imc)
    high_res_fake_imc = high_res_fake_imc.transpose((2, 0, 1))
    high_res_fake_imc = np.ascontiguousarray(high_res_fake_imc)
    high_res_fake_imc = torch.from_numpy(high_res_fake_imc).float().unsqueeze(0)
    high_res_fake_imc = high_res_fake_imc.to(device)
        
    if args.blur_sigma > 0:
        spatial_denoise = torchvision.transforms.GaussianBlur(3, sigma=args.blur_sigma)
        imc_roi_tensor_lvl2 = spatial_denoise(imc_roi_tensor_lvl2_raw)
        #high_res_fake_imc = spatial_denoise(high_res_fake_imc)
    

    # smooth predictions and ground truth, and flatten
    if args.avg_kernel>0:
        imc_roi_tensor_lvl2_smooth = avg_pool(imc_roi_tensor_lvl2)
        high_res_fake_imc_smooth = avg_pool(high_res_fake_imc)

    imc_roi_lvl2_smooth_flat = imc_roi_tensor_lvl2_smooth.detach().cpu().numpy()[0].transpose((1, 2, 0))
    imc_roi_lvl2_smooth_flat = imc_roi_lvl2_smooth_flat[:, :, [protein2index[prot_name] for prot_name in protein_set]].reshape((-1, len(protein_set)))
    high_res_fake_imc_smooth_flat = high_res_fake_imc_smooth.detach().cpu().numpy()[0].transpose((1, 2, 0)).reshape((-1, len(protein_set)))

    # prepare result structures:
    corrs = np.zeros(imc_roi_lvl2_smooth_flat.shape[1])
    pvals = np.zeros(imc_roi_lvl2_smooth_flat.shape[1])

    for chan_idx in range(imc_roi_lvl2_smooth_flat.shape[1]):
        (corrs[chan_idx], pvals[chan_idx]) = pearsonr(high_res_fake_imc_smooth_flat[:, chan_idx], imc_roi_lvl2_smooth_flat[:, chan_idx])

    for prot_idx, prot_name in enumerate(protein_set):
        ar["corr_" + prot_name] = corrs[prot_idx]
        ar["pval_" + prot_name] = pvals[prot_idx]

    dict_list.append(ar)
        
result_df = pd.DataFrame(dict_list)
result_df.to_csv(save_path_eval.joinpath('-'.join(['epoch'+str(epoch),'per_roi.csv'])))
print("Done !")




