import os 
import cv2
import numpy as np 
import pandas as pd
import openslide 
import glob
import argparse
import time 
import tqdm
import tifffile
import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import ImageDraw, Image
from collections import OrderedDict

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms

from shapely.geometry import Polygon, box
from shapely.affinity import scale
from shapely.ops import unary_union

from codebase.utils.constants import *
from codebase.utils.wsi_utils import *
from codebase.utils.inference_utils import *
from codebase.utils.eval_utils import get_last_epoch_number, get_protein_list
from codebase.experiments.cgan3.training_helpers import str2bool
from codebase.experiments.cgan3.network import * #Translator, Discriminator



parser = argparse.ArgumentParser(description='cGAN prediction on whole slide level')
parser.add_argument('--project_path', type=str, required=False, default= '/raid/sonali/project_mvs/', help='project base path')
parser.add_argument('--submission_id', type=str, required=True, help='experiment for which need to run wsi prediction')
parser.add_argument('--epoch', type=str, required=False, default='best', help='Which epoch to use (e.g.,2 or 2-3), if last then takes the last one found, if best then searching for chkpt_selection info')
parser.add_argument('--pth_name', type=str, required=False, default=None, help='Which epoch to use (pth file name), if "which_model" is old then provide desired model_state pth')
parser.add_argument('--wsi_paths', type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/HE_new_wsi', help='path where wsi h&e images reside')
parser.add_argument('--sample', type=str, required=False, default=None, help='sample name for which we need to run wsi prediction')
parser.add_argument('--which_cluster', type=str, required=True, help='biomed or dgx:gpu')
parser.add_argument('--set', type=str, required=False, default="test", help='Which set from split to use {test, valid, train}')
parser.add_argument('--batch_size', type=int, required=False, default=16, help='batch size for the data loader')
parser.add_argument('--ref_img_path', type=str, required=False, default=None, help='Path to reference image for stain normalization')

# ----- paths etc -----
args = parser.parse_args()
save_path = Path(args.project_path).joinpath('results')
model_path = Path(args.project_path).joinpath('results', args.submission_id, 'tb_logs')
CV_SPLIT_ROIS_PATH = Path(args.project_path).joinpath(CV_SPLIT_ROIS_PATH)
pth_name = args.pth_name
epoch = args.epoch
data_set = args.set
wsi_paths = args.wsi_paths

# ----- prepare reference img for stain normalization -----
normalizer = None
if args.ref_img_path:
    print("Initilize MacenkoNormalizer...")
    ref_img = Image.open(args.ref_img_path)
    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    normalizer.fit(tsfm(ref_img))

# ----- getting args.txt for experiment -----
job_args = json.load(open(Path(args.project_path).joinpath('results',args.submission_id,'args.txt')))

# ----- setting device -----
dev0 = 'cuda:0' if 'dgx' not in args.which_cluster else 'cuda:' + args.which_cluster.split(':')[1]
dev0 = torch.device(dev0 if torch.cuda.is_available() else 'cpu')
print('Device: ', dev0)

# ----- getting checkpoint pth file for which we run inference -----

if pth_name is None:
    if epoch=='best':
        chkpt_path = Path(str(model_path).replace('tb_logs', 'chkpt_selection'))
        # to allow to use different best checkpoints for performing snaity checks
        best_data_set = epoch.split('_')[-1] if '_' in epoch else 'valid'
        chkpt_file = '-'.join(['best_epoch','level_'+str(args.level),best_data_set])+'.txt'
        if os.path.exists(chkpt_path):
            pth_name = json.load(open(chkpt_path.joinpath(chkpt_file)))['chkpt_file']
            step_name = json.load(open(chkpt_path.joinpath(chkpt_file)))['best_step']
            print('Using best step: '+str(step_name))
        # else:
        #     print('Checkpoint selection data not found - using the last epoch found')
        #     epoch = 'last'
    elif epoch == 'last':
        print('Using the last step/epoch found.')
        step_name = get_last_epoch_number(model_path)
        pth_name = step_name + '_translator.pth'
    else: # eg when epoch='145K'
        step_name = str(epoch)
        pth_name = step_name + '_translator.pth'
else:
    step_name = str(pth_name.split('_')[0]) 

print('Analysing', args.submission_id, pth_name)  

# ----- loading model, setting to eval mode -----
network = torch.load(model_path.joinpath(pth_name), map_location=torch.device(dev0))
network.to(dev0)
network.eval()

# ----- get sample_roi names for a given CV split -----
if data_set=="external_test": 
    samples = np.array(list(set([i.split('.')[0] for i in os.listdir(wsi_paths)])))
else:
    cv_split = job_args['cv_split']
    cv = json.load(open(CV_SPLIT_ROIS_PATH))
    sample_rois = cv[cv_split][data_set]
    samples = np.array(list(set([i.split('_', 1)[0] for i in sample_rois]))) # getting sample names from sample_rois

# ----- defining specifics for inference -----
chunk_size = 4096 
chunk_padding = 0
batch_size = args.batch_size
loader_kwargs = {'num_workers': 8, 'pin_memory': True}
protein_subset = get_protein_list(job_args['protein_set'])  
print('protein_subset: ', len(protein_subset), protein_subset)

if args.sample == None: 
    print('samples: ', samples)
    save_path_imgs = save_path.joinpath(args.submission_id, data_set+"_wsis", "step_"+epoch)
    for sample in samples: 
        print('sample: ', sample)
        # Open the slide for reading
        print(wsi_paths, len(os.listdir(wsi_paths)))

        if (len(glob.glob(wsi_paths + '/' + sample + '*'))!=0):
            wsi_path = glob.glob(wsi_paths + '/' + sample + '*')[0]
            wsi = openslide.open_slide(wsi_path)
            seg_level = wsi.get_best_level_for_downsample(64) # level for tissue segementation
                    
            if not os.path.isfile(os.path.join(save_path_imgs, 'level_2', sample + '.npy')): 
                start_time = time.time()
                print(wsi)
                print(wsi.level_downsamples)
                print(wsi.level_dimensions)
                print(wsi.properties[openslide.PROPERTY_NAME_MPP_Y])

                get_wsi_inference(sample, wsi, seg_level, chunk_size, batch_size, len(protein_subset), 
                                loader_kwargs, dev0, network, save_path_imgs, normalizer)

                # timing
                end_time = time.time()
                print('time for wsi: ', end_time-start_time)
                hours, rem = divmod(end_time-start_time, 3600)
                minutes, seconds = divmod(rem, 60)
                print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
            else: 
                print('Inference already done for: ', sample)
        else: 
            print('WSI does not exist for: ', sample)


else: 
    # predict for only chosen sample
    sample = args.sample 
    print('sample: ', sample)
    
    # find this sample is in which dataset 
    with open(os.path.join(args.project_path, CV_SPLIT_SAMPLES_PATH), "r") as jsonFile:
        data = json.load(jsonFile)
    
    if sample in data[cv_split]['test']: 
        data_set = 'test'
    elif sample in data[cv_split]['valid']: 
        data_set = 'valid'
    elif sample in data[cv_split]['train']: 
        data_set = 'train'
    else: 
        data_set = 'external_test'
    print(data_set)
    save_path_imgs = save_path.joinpath(args.submission_id, data_set+"_wsis", "step_"+epoch)
   
    # Open the slide for reading
    if (len(glob.glob(wsi_paths + '/' + sample + '*'))!=0):
        wsi_path = glob.glob(wsi_paths + '/' + sample + '*')[0]
        wsi = openslide.open_slide(wsi_path)
        seg_level = wsi.get_best_level_for_downsample(64) # level for tissue segementation

        start_time = time.time()
        get_wsi_inference(sample, wsi, seg_level, chunk_size, batch_size, len(protein_subset), loader_kwargs, dev0, network, save_path_imgs)

        # timing
        end_time = time.time()
        print('time for wsi: ', end_time-start_time)
        hours, rem = divmod(end_time-start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    else: 
        print('no WSI image found for sample ', args.sample)