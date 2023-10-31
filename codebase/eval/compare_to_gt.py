import numpy as np
import pandas as pd
import os
from scipy.stats.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import math
import argparse
from scipy.ndimage import gaussian_filter

from codebase.utils.constants import *
from codebase.utils.metrics import *
from codebase.utils.eval_utils import *
from codebase.utils.viz_utils import *

from pathlib import Path
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Evaluation on a selected set.')
parser.add_argument('--save_path', type=str, required=False, default='/cluster/work/grlab/projects/projects2021-multivstain/results', help='Path to save the predictions')
parser.add_argument('--model_path', type=str, required=False, default='/cluster/work/grlab/projects/projects2021-multivstain/results', help='Path with model logs')
parser.add_argument('--set', type=str, required=False, default="test", help='Which set from split to use {test, valid, train}')
parser.add_argument('--model_type', type=str, required=False, default="cgan2", help='Which model to use')
parser.add_argument('--submission_id', type=str, required=True, default=None, help='Job submission id')
parser.add_argument('--epoch', type=int, required=False, default=None, help='Which epoch to use (number), if not specified the last computed is used')
parser.add_argument('--suppress_q', type=float, required=False, default=0.1, help='Quantile below which the signal is considered noise and is set to 0')
parser.add_argument('--standard_gt', type=bool, required=False, default=False, help='Whether to standardize GT instead of destandardize IMC')
parser.add_argument('--thrs_start', type=float, required=False, default=0, help='Starting value for binarization threshold grid')
parser.add_argument('--thrs_end', type=float, required=False, default=1, help='End value for binarization threshold grid')
parser.add_argument('--thrs_step', type=float, required=False, default=0.05, help='Step value for binarization threshold grid')
parser.add_argument('--thrs_how', type=str, required=False, default="quantile", help='Method for setting binarization threshold grid {step, quantile}')
parser.add_argument('--blur_sigma', type=int, required=False, default=0, help='Stdev the blurring Gaussian kernel (if 0, no blurring is applied)')

args = parser.parse_args()
save_path = Path(args.save_path)
img_path = Path(args.model_path).joinpath(args.submission_id,args.set+'_images')
epoch = args.epoch
if epoch is None:
    epoch = np.sort([int(x.split('epoch')[-1].split('.')[0]) for x in os.listdir(img_path)])[-1]

img_path = img_path.joinpath('epoch'+str(epoch))
files = os.listdir(img_path)

save_path_eval = save_path.joinpath(args.submission_id, args.set+'_eval')
if not os.path.exists(save_path_eval):
    save_path_eval.mkdir(parents=True, exist_ok=False)
save_path = save_path.joinpath(args.submission_id, args.set+'_plots')
if not os.path.exists(save_path):
    save_path.mkdir(parents=True, exist_ok=False)

# define subset of predicted proteins (based on args.txt):
job_args = pd.read_json(Path(args.model_path).joinpath(args.submission_id,'args.txt'), orient='index')
protein_set_name = job_args.loc['protein_set',:].values[0]
protein_set = get_protein_list(protein_set_name)

stand = 'stand' if args.standard_gt else 'destand'
    
for fname in files:
    sample_roi = fname.split('.')[0]
    print(sample_roi)
    img_gt = np.load(Path(BINARY_IMC_ROI_STORAGE).joinpath(fname))
    protein_idx_gt = [protein2index[protein] for protein in protein_set]
    img_gt = img_gt[:,:,protein_idx_gt]
    img_pred = np.load(img_path.joinpath(fname))
    if len(img_pred.shape)<3:
        img_pred = np.expand_dims(img_pred, axis=-1)
    
    if args.blur_sigma >0:
        img_gt = gaussian_filter(img_gt, sigma=args.blur_sigma)
        #img_pred = gaussian_filter(img_pred, sigma=args.blur_sigma)

    eval_df = pd.DataFrame(index=protein_set)
    for protein in protein_set:
        gt_idx = get_protein_idx(img_gt.shape[2], protein)
        img_gt_prot = img_gt[:,:,gt_idx]
        if args.standard_gt:
            img_gt = destandardize_img(img_gt, EXPRS_AVG[protein2index[protein]], EXPRS_STDEV[protein2index[protein]])
        img_gt_prot = np.apply_along_axis(suppress_to_zero,1,img_gt_prot, q=args.suppress_q)
        
        pred_idx = get_protein_idx(img_pred.shape[2], protein)
        img_pred_prot = img_pred[:,:,pred_idx]
        if args.standard_gt == False:
            img_pred_prot = destandardize_img(img_pred_prot, EXPRS_AVG[protein2index[protein]], EXPRS_STDEV[protein2index[protein]])
        img_pred_prot = np.apply_along_axis(suppress_to_zero,1,img_pred_prot, q=args.suppress_q)
        
        # compare signal distribution
        gt_df = pd.DataFrame({'values':img_gt_prot.flatten(),'type':'GT'})
        pred_df = pd.DataFrame({'values':img_pred_prot.flatten(),'type':'Pred'})
        plot_df = pd.concat([gt_df,pred_df], axis=0)
        sns.kdeplot(data=plot_df, x='values', hue='type', fill=True, alpha=0.4)
        save_fname = '-'.join([sample_roi,protein,'epoch'+str(args.epoch),protein_set_name,'gt_pred_distr','blur'+str(args.blur_sigma),'sup'+str(args.suppress_q).replace('.','_'),stand])+'.png'
        plt.savefig(save_path.joinpath(save_fname), dpi=300, bbox_inches='tight')
        plt.close()
        
        # compute correlation
        corr = pearsonr(gt_df['values'], pred_df['values'])[0]
        eval_df.loc[protein,'corr'] = corr
        
        # compute ssim
        ssim_score = ssim(np.uint8(100*img_gt_prot), np.uint8(100*img_pred_prot))
        eval_df.loc[protein,'ssim'] = ssim_score
        
        thrs_grid = get_thrs_grid(args.thrs_start,args.thrs_end,args.thrs_step, how=args.thrs_how, bin_vec=img_gt_prot.flatten())
        thrs_grid = [round(x,2) for x in thrs_grid]

        ncols = 5
        nrows = int(len(thrs_grid)/ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(20,14), constrained_layout=True)
        i = 0
        j = 0
        for thrs in thrs_grid:
            i = i % ncols 
            img_gt_bin = binarize_array(img_gt_prot, thrs)
            img_pred_bin = binarize_array(img_pred_prot, thrs)
            dice_score = dice(img_pred_bin, img_gt_bin)
            overlap = overlap_perc(img_pred_bin, img_gt_bin)
            plot_binary_overlap(img_gt_bin, img_pred_bin, plot_show=False, ax=axes[j,i], add_legend=False,
                               add_title='\n Thrs: '+str(round(thrs,2))) #, add_eval=False
            eval_df.loc[protein,'dice_'+str(thrs)] = dice_score
            eval_df.loc[protein,'overlap_'+str(thrs)] = overlap
            # Percentage positive pixels
            n_pixels = img_pred_bin.flatten().shape[0]
            eval_df.loc[protein,'pixelsGT_'+str(thrs)] = round(np.sum(img_gt_bin)/n_pixels,2)
            eval_df.loc[protein,'pixelsPred_'+str(thrs)] = round(np.sum(img_pred_bin)/n_pixels,2)
            i = i+1
            if i == ncols:
                j = j+1
        save_fname = '-'.join([sample_roi,protein,'epoch'+str(args.epoch),protein_set_name,'gt_pred_scatter','blur'+str(args.blur_sigma),args.thrs_how,'start'+str(args.thrs_start).replace('.','_'),'end'+str(args.thrs_end).replace('.','_'),'step'+str(args.thrs_step).replace('.','_'),'sup'+str(args.suppress_q).replace('.','_'),stand])+'.png'
        plt.savefig(save_path.joinpath(save_fname), dpi=300, bbox_inches='tight')
        plt.close()
 
    eval_df.index.name = 'protein'
    save_fname = '-'.join([sample_roi,'epoch'+str(epoch),protein_set_name,'gt_pred_eval','blur'+str(args.blur_sigma),args.thrs_how,'start'+str(args.thrs_start).replace('.','_'),'end'+str(args.thrs_end).replace('.','_'),'step'+str(args.thrs_step).replace('.','_'),'sup'+str(args.suppress_q).replace('.','_'),stand])+'.tsv'
    eval_df.to_csv(save_path_eval.joinpath(save_fname), sep='\t')
    
    # Plot metrics as a function of threshold
    eval_df = eval_df.drop(columns=['corr','ssim'])
    eval_df_long = eval_df.reset_index().melt(id_vars='protein', var_name='metric', value_name='value')
    eval_df_long['thrs'] = [np.float(x.split('_')[-1]) for x in eval_df_long['metric']]
    eval_df_long['metric'] = [x.split('_')[0] for x in eval_df_long['metric']]
    # map overlap to [0,1] interval
    eval_df_long.loc[eval_df_long.metric=='overlap','value'] = eval_df_long.loc[eval_df_long.metric=='overlap','value']/100
    for protein in protein_set:
        plot_df = eval_df_long.loc[eval_df_long.protein==protein,:]
        sns.lineplot(x='thrs', y='value', data=plot_df, hue='metric', legend=False)
        sns.scatterplot(x='thrs', y='value', data=plot_df, hue='metric')
        plt.title(protein)
        save_fname = '-'.join([sample_roi,'epoch'+str(epoch),protein_set_name,'thrs_metrics','blur'+str(args.blur_sigma),args.thrs_how,'start'+str(args.thrs_start).replace('.','_'),'end'+str(args.thrs_end).replace('.','_'),'step'+str(args.thrs_step).replace('.','_'),'sup'+str(args.suppress_q).replace('.','_'),stand])+'.png'
        plt.savefig(save_path.joinpath(save_fname), dpi=300, bbox_inches='tight')
        plt.close()


    