import numpy as np
import pandas as pd
from pathlib import Path
import os
import math
import json
import torch
import torchvision.transforms.functional as ttf
import torchvision.transforms as tt
import torch.nn as nn

from codebase.utils.constants import *
from codebase.utils.metrics import *

def get_resources_from_log(job_log=None):
    ''' Get information about computation (time and memory) from job log dataframe
    job_log: pandas dataframe with the job_log file loaded (using header=None and sep='\t')
             if None, then returns a dataframe with NaNs
    '''
    if job_log is None:
        df = pd.DataFrame(index=['CPU time  [sec.]', 'Max Memory  [MB]', 'Average Memory  [MB]',
       'Total Requested Memory  [MB]', 'Delta Memory  [MB]', 'Max Swap  [MB]',
       'Max Processes  [unknown]', 'Max Threads  [unknown]',
       'Run time  [sec.]'], columns = ['resource_val'])
        df.index.name = 'resource_type_unit'
    else:
        resource_idx = [i for i,x in enumerate(job_log.iloc[:,0]) if 'Resource' in str(x)][0]
        df = job_log.iloc[resource_idx+1:resource_idx+10,:]
        df.columns = ['resource']
        df[['resource_type', 'resource_val']] = df['resource'].str.split(':', 1, expand=True)
        df['resource_unit'] = [str(x).split(' ')[-1] for x in df['resource_val']]
        df['resource_unit'] = [x if x in ['sec.','MB'] else 'unknown' for x in df['resource_unit']]
        df['resource_val'] = [x.split(' ')[-2] if df['resource_unit'].iloc[i]!='unknown' else x.split(' ')[-1] for i,x in enumerate(df['resource_val'])]
        df['resource_val'] = [int(np.float(x)) if x!='-' else np.nan for x in df['resource_val']]
        df['resource_type_unit'] = [x.replace('    ','')+' ['+y+']' for x,y in zip(df['resource_type'], df['resource_unit'])]
        df = df.set_index('resource_type_unit')
        df = df.drop(columns=['resource', 'resource_unit','resource_type'])
    return df

def binarize_array(array, thrs=0):
    ''' Binarize an array (returns an array of the same shape with values in {0,1})
    array: array with numeric values
    thrs: threshold used for binarizing the array
    '''
    return 1*np.array(array>=thrs)


def get_thrs_grid(start=0, end=5, step=0.1, how='step', bin_vec=None):
    '''Generate a grid of thresholds for binarization
    start: minimal value (if how==quantile, must be within [0,1])
    end: maximal value (if how==quantile, must be within [0,1])
    step: step size (if how==quantile, must be within [0,1])
    how: how to compute thresholds {step, quantile}
    bin_vec: vector used to set threshold (if how in {median, quantile})
    '''
    assert start <= end, 'Start cannot be larger than end.'
    thrs_grid = np.arange(start, end, step)
    if np.isclose(start,end):
        thrs_grid = [start]
    if how=='quantile':
        assert bin_vec is not None, 'Missing bin_vec to calculate quantile.'
        thrs_grid = [np.quantile(bin_vec, q=q) for q in thrs_grid]
    return thrs_grid

# TODO: modify any code using this function as with new protein selectio set up it would not work!!
def get_protein_idx(n_channels, protein):
    ''' Obtain index corresponding to protein based on number of channels
    n_channels: number of channels {38,10,12,0}
    protein: protein name (see PROTEIN_LIST in constants for valid names)
    '''
    if n_channels==38:
        idx = protein2index[protein]
    elif n_channels==10:
        idx = [i for i,x in enumerate(celltype_relevant_prots)][0]
    elif n_channels==12:
        idx = [i for i,x in enumerate(celltype_relevant_prots_ext)][0]
    elif n_channels==5:
        idx = [i for i,x in enumerate(immune_prots)][0]
    elif n_channels==2:
        idx = [i for i,x in enumerate(tumor_prots)][0]
    elif n_channels==1:
        idx = 0
    return idx

def get_protein_list(protein_set_name):
    '''Get list of protein names correspondig to protein_set_name
    protein_set_name: Name of the protein set {full, reduced, reduced_ext, single_protein_name}
    '''
    if protein_set_name=='full':
        protein_set = PROTEIN_LIST  # predict all proteins
    elif protein_set_name=='reduced':
        protein_set = celltype_relevant_prots # predict 10 proteins as in report
    elif protein_set_name=='selected':
        protein_set = selected_prots # predict 10 proteins, replaced SOX10 with SOX9 in reduced set 
    elif protein_set_name=='selected_snr':
        protein_set = selected_prots_snr # predict 11 proteins, chosen by Ruben on 03.02.23
    elif protein_set_name=='selected_snr_dgm4h':
        protein_set = selected_prots_snr_dgm4h # predict 10 proteins, same as miccai but w/o glut1; used for dgm4h
    elif protein_set_name=='selected_snr_v2':
        protein_set = selected_prots_snr_v2 # predict 12 proteins, replaced GLUT1 with ki-67 (proliferation marker) from selected_prots_snr and added SMA
    elif protein_set_name=='prots_pseudo_multiplex':
        protein_set = prots_pseudo_multiplex # predict 10 proteins, for miccai
    elif protein_set_name=='prots_tls':
        protein_set = prots_tls # predict all proteins for tls 
    elif protein_set_name=='prots_cd16_correlates':
        protein_set = prots_cd16_correlates # predict all proteins for cd16 
    elif protein_set_name=='prots_ki67_correlates':
        protein_set = prots_ki67_correlates # predict all proteins for Ki-67 
    elif protein_set_name=='selected_snr_ext':
        protein_set = selected_prots_snr_ext # predict 16 proteins, chosen by Ruben on 03.02.23
    elif protein_set_name=='reduced_ext':
        protein_set = celltype_relevant_prots_ext # predict extended list of proteins
    elif protein_set_name=='reduced_ext_ir':
        protein_set = celltype_relevant_prots_ext_ir # predict extended list of proteins + Ir
    elif protein_set_name == 'tumor_prots':
        protein_set = tumor_prots
    elif protein_set_name == 'immune_prots':
        protein_set = immune_prots
    elif protein_set_name == 'tcell_prots':
        protein_set = tcell_prots
    elif protein_set_name == 'ir_cd8':
        protein_set = ir_cd8
    elif protein_set_name == 'tumor_cd8':
        protein_set = tumor_cd8
    elif protein_set_name == 'cd3_cd8':
        protein_set = cd3_cd8
    elif protein_set_name == 'tumor_cd8_cd3':
        protein_set = tumor_cd8_cd3
    else:
        protein_set = [protein_set_name] # predict a single protein
    return protein_set


def arcsinh_trafo(x):
    ''' Arcsinh trasformation of values in array (1 or more-dimensional)
    '''
    if len(x)>1:
        return pd.Series(x).apply(lambda x: math.log(x + math.sqrt(x**2 + 1)))
    else:
        return math.log(x + math.sqrt(x**2 + 1))
    
def standardize_img(img, global_avg=0, global_stdev=1):
    '''Center and standardize image using global_avg and global_stdev
    '''
    return (img - global_avg)/global_stdev

def destandardize_img(img, global_avg=0, global_stdev=1):
    '''Revert standardization and centering of the image using global_avg and global_stdev
    '''
    return (img*global_stdev) + global_avg


def min_max_scale(x, min_cohort, max_cohort):
    return (x-min_cohort)/(max_cohort - min_cohort)

def standardize(x, mean_cohort, std_cohort):
    return (x-mean_cohort)/std_cohort


def suppress_to_zero(x, q=0.25):
    ''' Set value considered noise (below quantile q) to 0
    x: vector with values
    q: quantile [0,1]
    '''
    thrs = np.quantile(x,q)
    x[x<thrs] = 0
    return x

def get_events_path(root, discriminator=True):
    ''' Get path to tensorboard events file
    root: absolute path to tb_logs directory
    '''
    root = Path(root)
    keyword = 'losses_discriminator' if discriminator else 'losses_translator'
    subdirs = [subdir for subdir in os.listdir(root) if keyword in subdir]
    assert len(subdirs)>0, 'No directory with losses found.'
    fnames = [fname for fname in os.listdir(root.joinpath(subdirs[0])) if 'events' in fname]
    assert len(fnames)>0, 'No event file with losses found.'

    return str(root.joinpath(subdirs[0], fnames[0]))

def get_loss_from_events(events_path, loss_name='val_loss', per_epoch=True, steps_per_epoch=20):
    ''' Load loss values from tensorboard events file
    events_path: absolute path to events file
    loss_name:
    '''
    # based on https://stackoverflow.com/questions/42355122/can-i-export-a-tensorflow-summary-to-csv
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    event_accumulator = EventAccumulator(events_path)
    event_accumulator.Reload()

    loss_tag = event_accumulator.Tags()['scalars'][0]
    steps = {x.step for x in event_accumulator.Scalars(loss_tag)}
    x = list(range(len(steps)))
    y = [x.value for x in event_accumulator.Scalars(loss_tag) if x.step in steps]
    df = pd.DataFrame({"step": x, loss_name: y})
    if per_epoch:
        df['epoch'] = df.step.apply(lambda x: 'epoch'+str(int(np.floor(x / steps_per_epoch))))
        df = df.drop(columns='step').groupby('epoch').median()
        df = df.reset_index(drop=False)

    return df


def select_checkpoint(eval_df, weight_dict, ntop=1):
    ''' Select the best checkpoint based on weighted average of eval metrics
    eval_df: pd dataframe with columns {corr,ssim,dice,val_loss} and epoch names in index
    weight_dict: dictionry with columns form eval_df and corresponding weights
    ntop: number of best epochs to report
    '''
    cols = [x for x in weight_dict.keys()]
    assert np.isclose(len(set(cols).intersection(eval_df.columns)), len(cols)), print('Some metrics from weight_dict are missing in eval_df!')
    eval_df = eval_df.loc[:,cols]
    weight_vec = [x for x in weight_dict.values()]
    epoch_mean = eval_df.transpose().apply(lambda x: np.mean(weight_vec*x))

    return epoch_mean.nlargest(ntop).index.to_list()

def get_ordered_epoch_names(model_path):
    ''' Function to return names of modelstate files ordered by the epoch (usual sorting function does not properly sort bcs of the epoch naming convention)
    model_path: absolute path to the tb_logs folder of a given job
    '''
    files = [x for x in os.listdir(model_path) if 'model_state' in x]
    df_steps = pd.DataFrame({'step_number':[int(x.split('K_')[0]) for x in files], 'step_name':[x.split('_')[0] for x in files]})
    df_steps = df_steps.sort_values(by=['step_number'], ascending=True)
    return df_steps.step_name.to_list()


def get_last_epoch_number(model_path):
    ''' Function to find the last epoch for a given job
    model_path: absolute path to the tb_logs folder of a given job
    '''
    ordered_step_names = get_ordered_epoch_names(model_path)
    return ordered_step_names[-1]


def get_best_epoch_w_imgs(project_path, submission_id, level=2, data_set='valid'):
    ''' Extract the best epoch from chkpt selection files and check if corresponding inferred images exist
    project_path: Path where all data, results etc for project reside
    submission_id: job submission id
    level: resolution level ({2,4,6})
    data_set: data set {train, valid, test}
    '''
    if isinstance(project_path, str):
        project_path = Path(project_path)
    chkpt_path = project_path.joinpath('results', submission_id, 'chkpt_selection')
    if os.path.exists(chkpt_path):
        chkpt_file = '-'.join(['best_epoch','level_'+str(level),data_set])+'.txt'
        best_step_name = json.load(open(chkpt_path.joinpath(chkpt_file)))['best_step']
    if os.path.exists(project_path.joinpath('results', submission_id, data_set+'_images', 
                                            'step_' + best_step_name, 'level_'+str(level))):
        return best_step_name
    else:
        print('No epoch with images found')
        return np.nan


def get_last_epoch_w_imgs(project_path, submission_id, level=2, data_set='valid'):
    ''' Find the last epoch for which corresponding inferred images exist
    project_path: Path where all data, results etc for project reside
    submission_id: job submission id
    level: resolution level ({2,4,6})
    data_set: data set {train, valid, test}
    '''
    if isinstance(project_path, str):
        project_path = Path(project_path)
    imgs_path = project_path.joinpath('results', submission_id, data_set+'_images')
    imgs_steps = [x for x in os.listdir(imgs_path)]
    df_steps = pd.DataFrame({'step_number':[int(x.split('_')[1].split('K')[0]) for x in imgs_steps], 'step_name':[x.split('_')[1] for x in imgs_steps]})
    df_steps = df_steps.sort_values(by=['step_number'], ascending=True)
    last_step_name = df_steps.step_name.to_list()[-1]

    if os.path.exists(project_path.joinpath('results', submission_id, data_set+'_images', 'step_' + last_step_name, 'level_'+str(level))):
        return last_step_name
    else:
        print('No epoch with images found')
        return np.nan
    
def convert_np_to_df(img, img_type='GT'):
    ''' Convert numpy 1 channel image to long format pandas df
    img: numpy array with max 1 channel
    img_type: which label to give the image, eg GT, pred
    '''
    img_df = pd.DataFrame(img.reshape(img.shape[0], img.shape[1]),
                            index=[str(x) for x in range(img.shape[0])],
                            columns=[str(x) for x in range(img.shape[1])])
    img_df.index.name = 'y_coords'
    img_df = img_df.reset_index(drop=False)
    img_df = img_df.melt(id_vars='y_coords', var_name='x_coords', value_name='value')
    img_df = img_df.loc[img_df['value']>0,:]
    img_df['y_coords'] = img_df['y_coords'].astype(int)
    img_df['x_coords'] = img_df['x_coords'].astype(int)
    img_df['img_type'] = img_type
    
    return img_df


def preprocess_img(img, dev0, downsample_factor=1, kernel_width=32, blur_sigma=0, avg_kernel=32, avg_stride=1):
    ''' Function to preprocess an img for evaluations
    img: np array tensor [H,W,C]
    dev0: device to send the data for torch operations
    downsample_factor: if > 1 then downsamples the img to shape[0]//downsample_factor
    kernel_width: Gaussian kernel size
    blur_sigma: Gaussian kernl sigma
    avg_kernel: averaging square kernel size
    avg_stride: averaging kernel stride
    '''
    img = torch.from_numpy(img.copy().transpose(2,0,1))
    # send to device for performing torch operations
    img.to(dev0)
    # downsample GT if pred IMC downsampled
    if downsample_factor > 1:
        img = ttf.resize(img, img.shape[1]//downsample_factor)
    if avg_kernel>0:
        avg_pool = nn.AvgPool2d(kernel_size=avg_kernel, padding=int(np.floor(avg_kernel/2)), stride=avg_stride)
        img = avg_pool(img)
    if blur_sigma > 0:
        spatial_denoise = tt.GaussianBlur(kernel_width, sigma=blur_sigma)
        img = spatial_denoise(img)
    img = np.asarray(img).transpose(1,2,0)
    
    return img


def get_thrs_metrics_protein(img_gt, img_pred, dice_score=True, overlap=False, perc_pos=False, densitycorr=False,
                                thrs_start=0, thrs_end=1, thrs_step=0.1, thrs_how='quantile', thrs_cohort_df=None, densitycorr_px=32, densitycorr_metric='pcorr'):
    ''' Get threshold-based eval metrics for one protein
    img_gt: numpy array [H,W] with ground truth protein expression
    img_pred: numpy array [H,W] with predicted protein expression
    dice_score: whether to compute Dice score
    overlap: whether to compute overlap_percentage
    perc_pos: whether to compute percentage of positive pixels
    densitycorr: whether to compute density-based correlation
    densitycorr_px: desired resolution in px to compute density; n_bins=1000//densitycorr_px
    densitycorr_metric: which correlation coefficient to use to compare densities {pcorr, spcorr}
    '''
    if thrs_how=='cohort':
        thrs_grid = [np.float(x.replace('q', '')) for x in thrs_cohort_df.index.to_list()]
        thrs_vals_grid_gt = thrs_cohort_df.iloc[:,0].to_list()
        thrs_vals_grid_pred = thrs_cohort_df.iloc[:,0].to_list()
    else:
        # get the grid of quantiles used for signal binarization
        thrs_grid = [thrs_start] if np.isclose(thrs_start, thrs_end) else np.arange(thrs_start, thrs_end, thrs_step)
        # get the corresponding grid of threshold values for GT and pred
        thrs_vals_grid_gt = get_thrs_grid(thrs_start, thrs_end, thrs_step, how=thrs_how, bin_vec=img_gt.flatten())
        #thrs_vals_grid_gt = [round(x,2) for x in thrs_vals_grid_gt]
        thrs_vals_grid_pred = get_thrs_grid(thrs_start, thrs_end, thrs_step, how=thrs_how, bin_vec=img_pred.flatten())
        #thrs_vals_grid_pred = [round(x,2) for x in thrs_vals_grid_pred]

    thrs_eval_df = pd.DataFrame(index=['protein'])
    for j, thrs in enumerate(thrs_grid):
        thrs_name = str(round(thrs_grid[j],2))
        # binarize arrays
        img_gt_bin = np.zeros(img_gt.shape) if np.isclose(thrs_vals_grid_gt[j],0) else binarize_array(img_gt, thrs_vals_grid_pred[j])
        img_pred_bin = np.zeros(img_pred.shape) if np.isclose(thrs_vals_grid_pred[j],0) else binarize_array(img_pred, thrs_vals_grid_pred[j])
        # compute metrics
        if dice_score:
            thrs_eval_df.loc['protein','dice_'+thrs_name] = dice(img_pred_bin, img_gt_bin)
        if overlap:
            thrs_eval_df.loc['protein','overlap_'+thrs_name] = overlap_perc(img_pred_bin, img_gt_bin)
        if density_corr:
            thrs_eval_df.loc['protein','densitycorr_'+thrs_name] = density_corr(img_gt_bin, img_pred_bin, metric=densitycorr_metric, desired_resolution_px=densitycorr_px, bin_lim=None, axmax=None)
        if perc_pos:
            thrs_eval_df.loc['protein','pixelsGT_'+thrs_name] = round(np.sum(img_gt_bin)/(img_gt_bin.shape[0]*img_gt_bin.shape[1]),2)
            thrs_eval_df.loc['protein','pixelsPred_'+thrs_name] = round(np.sum(img_pred_bin)/(img_pred_bin.shape[0]*img_pred_bin.shape[1]),2)
    
    return thrs_eval_df

def get_tumor_prots_signal(img_np, protein_list, tumor_prots=['MelanA', 'gp100', 'S100', 'SOX9', 'SOX10']):
    ''' Aggregate tumor signal by taking max across tumor markers
    img_np: numpy array with protein expression [H,W,C]
    protein_list: protein_list corresponding to the columns of img_np (in the same order)
    tumor_prots: tumor proteins to use for aggregation (only thos found in protein_list will be used!)
    '''
    sel_prots = [x for x in tumor_prots if x in protein_list]
    assert len(sel_prots)>0, 'No tumor prots found!'
    sel_prots_idx = [protein_list.index(prot) for prot in sel_prots]
    img_np_sel = img_np[:,:,sel_prots_idx]
    # take max across all tumor markers
    img_np_sel = np.apply_along_axis(np.max, 2, img_np_sel).reshape(img_np_sel.shape[0], img_np_sel.shape[1], 1)
    
    return img_np_sel