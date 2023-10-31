import os
from datetime import datetime
import pandas as pd
import numpy as np
import argparse
import random
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import torchvision.transforms.functional as ttf
from torch.optim import lr_scheduler
from torch.nn import init
import kornia

from torchinfo import summary



from codebase.utils.dataset_utils import clean_train_val_test_sep_for_manual_aligns_ordered_by_quality, filter_and_transform
#from codebase.utils.cv_split import kfold_splits
from codebase.utils.constants import *


def get_aligns(project_path, cv_split=None, protein_set='full', aligns_set='train'):
    ''' Return a list of aligns for train and valid (dictionary per sample_ROI)
    cv_split: name of the cv split {split1, ..., split5}; if None then report set is used
    good_areas_only: whether to suset to good areas only
    protein_set: name of the protein set (special tratment of 'reduced')
    aligns_set: set of aligns to be returned {train, valid, test}
    Note: needs refactoring; requires constants.py to be sources
    '''
    aligns_order = dict({'train':0, 'valid':1, 'test':2})
    if cv_split is None:
        aligns = clean_train_val_test_sep_for_manual_aligns_ordered_by_quality()
        aligns = aligns[aligns_order[aligns_set]]
     
    else:
        kfold_splits = json.load(open(os.path.join(project_path, CV_SPLIT_DICT_PATH)))
        assert cv_split in [x for x in kfold_splits.keys()], 'Selected cv split not in the kfold_splits'
        aligns = kfold_splits[cv_split][aligns_set]
        if protein_set=='reduced':
            aligns = [ar for ar in aligns if not ar["celltype_relevant_proteins_missing"]]

    return aligns



def add_noise(tensor, factor=0.1):
    ''' Add noise to a tensor
    tensor: tensor object
    factor: factor determining the amount of noise
    '''
    # as per Simon's code factor* is not sent, only torch.rand is sent (see cgan2, seems incorrect)
    tensor_noise = factor * torch.rand(tensor.size())
    if tensor.device != tensor_noise.device:
        # if input tensor is on device, need to send noise to the same device to perform addition
        tensor_noise = tensor_noise.to(tensor.device)
    tensor_w_noise = tensor + tensor_noise
    tensor_w_noise = tensor_w_noise.to(tensor.device)

    return tensor_w_noise

def add_noise_prob(tensor, factor=0.1, p=0.5):
    ''' Add noise to a tensor
    tensor: tensor object
    factor: factor determining the amount of noise
    '''
    if random.random() < p:
        # as per Simon's code factor* is not sent, only torch.rand is sent (see cgan2, seems incorrect)
        tensor_noise = factor * torch.rand(tensor.size())
        if tensor.device != tensor_noise.device:
            # if input tensor is on device, need to send noise to the same device to perform addition
            tensor_noise = tensor_noise.to(tensor.device)
        tensor_w_noise = tensor + tensor_noise
        tensor_w_noise = tensor_w_noise.to(tensor.device)
    else: 
        tensor_w_noise = tensor

    return tensor_w_noise

def resize_tensor(tensor, output_size):
    ''' Resize tensor to given output_size(s)
    tensor: tensor object
    output_size: scalar or list of output_sizes (resizing sequentially)
    returns resized tensors
    '''
    if isinstance(output_size, list) == False:
        output_size = [output_size]
    tensors_resized = []
    for osz in output_size:
        tensor = ttf.resize(tensor, osz)
        tensors_resized.append(tensor)

    return [x for x in tensors_resized]

# based on BCI paper: code: https://github.com/bupt-ai-cz/BCI/blob/55e0500dd3b9bd025c1e46f835c1335f2b10cc72/PyramidPix2pix/models/pix2pix_model.py
def scale_transform(tensor_img): 
    # apply 4 times gaussian blur and the downsample
    for i in range(4): 
        tensor_img = kornia.filters.gaussian_blur2d(tensor_img, (3,3), (1,1))
    tensor_img = kornia.filters.blur_pool2d(tensor_img, 1, stride=2)
    return tensor_img


# from https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/2
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def get_datetime():
    datetime_full_iso = datetime.today().isoformat().replace(":", "")
    datetime_full_iso = datetime_full_iso[:datetime_full_iso.rfind(".")]
    return datetime_full_iso

def str2bool(v):
    ''' Function alloowing for more flexible use of boolean arguments; 
    from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')