import os
from datetime import datetime
import pandas as pd
import numpy as np
import json
import math
import random
import cv2
import argparse
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
import time
import resource
import memory_profiler
import pickle

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
import torch.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity

from torchinfo import summary

from codebase.utils.constants import *
from codebase.utils.raw_utils import *
from codebase.utils.dataset_utils import clean_train_val_test_sep_for_manual_aligns_ordered_by_quality, filter_and_transform
from codebase.utils.cv_split import kfold_splits
from codebase.utils.zipper import do_zip
from codebase.utils.eval_utils import get_protein_list

from codebase.experiments.cgan3.network import Translator, Discriminator
from codebase.experiments.cgan3.loaders import CGANDataset

def get_aligns(cv_split=None, good_areas_only=True, precise_only=False, protein_set='full', aligns_set='train'):
    ''' Return a list of aligns for train and valid (dictionary per sample_ROI)
    cv_split: name of the cv split {split1, ..., split5}; if None then report set is used
    good_areas_only: whether to suset to good areas only
    protein_set: name of th eprotein set (special tratemnt of 'reduced')
    aligns_set: set of aligns to be returned {train, valid, test}
    Note: needs refactoring; requires constants.py to be sources
    '''
    aligns_order = dict({'train':0, 'valid':1, 'test':2})
    if cv_split is None:
        aligns = clean_train_val_test_sep_for_manual_aligns_ordered_by_quality()
        aligns = aligns[aligns_order[aligns_set]]
        if good_areas_only:
            aligns = [ar for ar in aligns if ar["precise_enough_for_l1"] or len(ar["good_areas"]) > 0]
        if precise_only:
            aligns = [ar for ar in aligns if ar["precise_enough_for_l1"]]
    else:
        assert cv_split in [x for x in kfold_splits.keys()], 'Selected cv split not in the kfold_splits'
        aligns = kfold_splits[args.cv_split][aligns_set]
        if good_areas_only:
            aligns = [ar for ar in aligns if (ar["precise_enough_for_l1"] or len(ar["good_areas"]) > 0)]
        if precise_only:
            aligns = [ar for ar in aligns if ar["precise_enough_for_l1"]]
        if protein_set=='reduced':
            aligns = [ar for ar in aligns if not ar["celltype_relevant_proteins_missing"]]

    return aligns


def get_optimizer(network, network_type, lr_scheduler_type='fixed'):
    ''' Get optimizer for network
    network: network object
    network_type: type of the network {discriminator, translator}
    lr_scheduler_type: type of the learning rate scheduler {fixed, plateau, cosine, step}
    '''
    if lr_scheduler_type == 'fixed':
        lr_val = 0.0008 if network_type=='discriminator' else 0.004
    else:
        lr_val = 0.002
    optimizer = optim.Adam(network.parameters(), lr=lr_val, betas=(0.5, 0.999))

    return optimizer

def get_lr_scheduler(optimizer, lr_scheduler_type='fixed', **kwargs):
    ''' Get learning rate schedulers
    optimizer: optimizer object
    lr_scheduler_type: type of the learning rate scheduler {fixed, plateau, cosine, step}
    '''
    if lr_scheduler_type == 'fixed':
        network_lr_scheduler = None
    else:
        if lr_scheduler_type == 'plateau':
            network_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, threshold=0.0001, patience=100)
        elif lr_scheduler_type == 'cosine':
            network_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)
        elif lr_scheduler_type == 'step':
            network_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    return network_lr_scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights. 
    From: https://gitlab.com/eburling/SHIFT/-/blob/master/models/networks.py

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


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

def update_translator_bool(rule='prob', dis_fake_loss=None):
    ''' Get bool whether to update translator
    rule: rule for updating {always, prob, dis_loss}
    dis_fake_loss: fake_score_mean value from discriminator (used only if rule==dis_loss)
    '''
    if rule == 'always':
        if_update = True
    elif rule == 'prob':
        if_update = random.choice([True, True, False])
    elif rule == 'dis_loss':
        assert dis_fake_loss is not None, 'dis_fake_loss not provided!'
        # TODO: check if 0.5 threshold makes sense
        if_update = dis_fake_loss<0.5

    return if_update

def get_datetime():
    datetime_full_iso = datetime.today().isoformat().replace(":", "")
    datetime_full_iso = datetime_full_iso[:datetime_full_iso.rfind(".")]
    return datetime_full_iso

def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_params(model):
    for param in model.parameters():
        param.requires_grad = True

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
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cGAN.')
    parser.add_argument('--no_zip', action='store_true', help='By default, the script saves the state of the code base as a timestamped zip file. Setting this flag turns it off.')
    parser.add_argument('--verbose', type=int, required=False, default=0, help='Verbose level')
    parser.add_argument('--seed', type=int, required=False, default=0, help='Random seed')
    parser.add_argument('--epochs', type=int, required=False, default=50, help='Number of epochs to train')
    parser.add_argument('--save_path', type=str, required=False, default=None, help='Path to save the log files and results')
    parser.add_argument('--cv_split', type=str, required=False, default=None, help='Selected CV split, if None then the splitting used for the report is used')
    parser.add_argument('--protein_set', type=str, required=False, default='full', help='which protein set to use')
    parser.add_argument('--restore_translator', type=str, required=False, default="NO_FILE", help='Translator checkpoint to restore the model from')
    parser.add_argument('--restore_discriminator', type=str, required=False, default="NO_FILE", help='Discriminator checkpoint to restore the model from')
    parser.add_argument('--dis_lr_scheduler', type=str, required=False, default='fixed', help='Learning rate scheduler for discriminator network {step, plateau, cosine}')
    parser.add_argument('--trans_lr_scheduler', type=str, required=False, default='fixed', help='Learning rate scheduler for translator network {step, plateau, cosine}')
    parser.add_argument('--trans_update_rule', type=str, required=False, default='prob', help='Rule for updating translator {always, prob, dis_loss}')
    parser.add_argument('--good_areas_only', type=str2bool, required=False, default=False, help='Whether to subset data to well-aligned only')
    parser.add_argument('--trans_weights_init', type=str, required=False, default='xavier', help='Weights initialization for translator')
    parser.add_argument('--dis_weights_init', type=str, required=False, default='xavier', help='Weights initialization for discriminator')
    parser.add_argument('--precise_only', type=str2bool, required=False, default=False, help='Whether to subset data to well-aligned whole ROIs only')
    parser.add_argument('--blur_gt', type=str2bool, required=False, default=True, help='Whether to blur ground-truth IMC data')
    parser.add_argument('--dis_add_noise', type=str2bool, required=False, default=False, help='Whether to add noise for training discriminator')
    parser.add_argument('--submission_id', type=str, required=False, default='tmp', help='Job submission ID')
    parser.add_argument('--patch_size', type=int, required=False, default=256, help='Patch size')
    parser.add_argument('--batch_size', type=int, required=False, default=16, help='Batch size')
    parser.add_argument('--save_every_x_epoch', type=int, required=False, default=20, help='Save logs every 1/x epoch.')
    
    args = parser.parse_args()
    # TODO: add name of the current file to args output to trace which script was executed (until we converge to only one script)
    # https://stackoverflow.com/questions/4152963/get-name-of-current-script-in-python
    t1_start = time.perf_counter()
    t2_start = time.process_time()

    # set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # get surrounding folder to store logs etc:
    if args.save_path is None:
        surrounding_path = os.path.realpath(__file__)
        surrounding_path = surrounding_path[:surrounding_path.rfind(os.path.sep)]
    else:
        surrounding_path = args.save_path

    # get current datetime up to seconds to create log folder:
    # TODO: maybe better to stick to submission_id only?
    datetime_full_iso = get_datetime()
    log_folder = os.path.join(surrounding_path, "tb_logs")

    if not os.path.exists(log_folder):
        Path(log_folder).mkdir(parents=True)
    writer = SummaryWriter(log_folder)

    if not args.no_zip:
        zip_folder = os.path.join(surrounding_path, "code_zips")
        if not os.path.exists(zip_folder): 
            Path(zip_folder).mkdir(parents=True)
        do_zip(label=datetime_full_iso, save_folder=zip_folder)
        
    with open(os.path.join(surrounding_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print('Created output folders')
    
    with profiler.profile(with_stack=True, profile_memory=True, on_trace_ready=profiler.tensorboard_trace_handler(os.path.join(log_folder)), activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], schedule=profiler.schedule(wait=5, warmup=5, active=3, repeat=1)) as prof:


        protein_subset = get_protein_list(args.protein_set)    
        train_aligns = get_aligns(cv_split=args.cv_split, good_areas_only=args.good_areas_only, precise_only=args.precise_only, protein_set=args.protein_set, aligns_set='train')
    #     train_aligns = train_aligns[:4]

        batch_size = args.batch_size
        # TODO: atm only patch size of 256 works!!
        patch_size = args.patch_size #256 #320  # µms, should be a multiple of 16
        assert patch_size==256, 'Atm patch size of 256 is the only implemented option.'
        print('Read all args.')

    #     print('Train set with '+str(len(train_aligns))+' ROIs')
    #     train_ds = CGANDataset(align_results=train_aligns,
    #                            name="Train",
    #                            patch_size=patch_size,
    #                            protein_subset=protein_subset,
    #                            verbose=2,
    #                            good_areas_only=args.good_areas_only)

        print('WIP: running on 3 ROIs only for debugging!')
    #     with open('/cluster/work/grlab/projects/projects2021-multivstain/data/tupro/debug_set/v0.pickle','wb') as handle:
    #         pickle.dump(train_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle = open('/cluster/work/grlab/projects/projects2021-multivstain/data/tupro/debug_set/v0.pickle','rb')
        train_ds = pickle.load(handle)
        print('Loaded data')
#         # TODO: increase number of ROIs (temporary decrease for code profiling only)
#         train_ds = train_ds[:1]
        trainloader = DataLoader(train_ds,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True, # for faster copying between devices (https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
                                 num_workers=1, # bcs of warning on ethsec (previously 8)
                                 drop_last=True)  # don't want any crashes with the last batch

        dev0 = torch.device('cuda:0')
        dev1 = torch.device('cuda:1')

        # create models
        if args.restore_translator == "NO_FILE":
            trans = Translator(num_imc_channels=len(protein_subset))
            init_weights(trans, init_type=args.trans_weights_init, init_gain=0.02)
            epoch_number = 0
        else:
            # restore model from checkpoint
            print('Restoring translator weights from a checkpoint.')
            trans = torch.load(args.restore_translator)
            epoch_number = int(str(args.restore_translator).split('_')[-1].split('.')[0])+1

        trans.to(dev0)


        if args.restore_discriminator == "NO_FILE":
            dis = Discriminator(num_imc_channels=len(protein_subset))
            init_weights(dis, init_type=args.dis_weights_init, init_gain=0.02)
        else:
            # restore model from checkpoint
            print('Restoring discriminator weights from a checkpoint.')
            dis = torch.load(args.restore_discriminator)
        dis.to(dev1)

        dis_opti = get_optimizer(dis, 'discriminator', args.dis_lr_scheduler)
        dis_lr_scheduler = get_lr_scheduler(dis_opti, args.dis_lr_scheduler)
        trans_opti = get_optimizer(trans, 'translator', args.trans_lr_scheduler)
        trans_lr_scheduler = get_lr_scheduler(trans_opti, args.trans_lr_scheduler)

        if args.blur_gt:
            spatial_denoise = torchvision.transforms.GaussianBlur(3, sigma=1)

        unfreeze_params(dis)
        unfreeze_params(trans)
        trans.train()
        dis.train()

        print('Initalization finished')

        for epoch_idx in range(epoch_number, args.epochs+epoch_number):
            print('At epoch: ',epoch_idx)

            # loss metrics for logging
            # TODO: why is this needed?
            dis_loss_val, trans_loss_val = 0.0, 0.0

            for i, batch in enumerate(trainloader):
                print(i)

                lv2_real_imc = batch["imc_patch"].to(dev0)
                # TODO: why there are two separate (maybe bcs we send them to two different devices due to memory issues)
                lv0_he_dev0 = batch["he_patch"].to(dev0)
                lv0_he_dev1 = batch["he_patch"].to(dev1)

                # prepare different resolutions of IMC and HE
                with torch.no_grad():
                    if args.blur_gt:
                        lv2_real_imc = spatial_denoise(lv2_real_imc)
                    lv4_real_imc, lv6_real_imc = resize_tensor(lv2_real_imc, [patch_size//4, patch_size//16])

                    lv2_he_dev0, lv4_he_dev0, lv6_he_dev0 = resize_tensor(lv0_he_dev0, [patch_size, patch_size//4, patch_size//16])
                    lv2_he_dev1, lv4_he_dev1, lv6_he_dev1 = resize_tensor(lv0_he_dev1, [patch_size, patch_size//4, patch_size//16])

                ##### do a step for the discriminator #####
                # to avoid mixing gradients across minibatches (.backward() accumulates gradients!)
                # none for memory saving, see https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
                dis_opti.zero_grad(set_to_none=True)
                #print('Updating discriminator')
                freeze_params(trans)
                with torch.no_grad():
                    lv2_fake_imc, lv4_fake_imc, lv6_fake_imc = trans(lv0_he_dev0, lv2_he_dev0, lv4_he_dev0, lv6_he_dev0)

                fake_tensors = [lv2_fake_imc.to(dev1), lv4_fake_imc.to(dev1), lv6_fake_imc.to(dev1)]
                real_tensors = [lv2_real_imc.to(dev1), lv4_real_imc.to(dev1), lv6_real_imc.to(dev1)]
                if args.dis_add_noise:
                    # make IMC noisy to make things harder for discriminator
                    #(maybe should skip this step; based on discussion with Gunnar - thus made optional for now)
                    noisy_lv2_fake_imc, noisy_lv4_fake_imc, noisy_lv6_fake_imc = [add_noise(x) for x in fake_tensors]
                    noisy_lv2_real_imc, noisy_lv4_real_imc, noisy_lv6_real_imc = [add_noise(x) for x in real_tensors]
                else:
                    noisy_lv2_fake_imc, noisy_lv4_fake_imc, noisy_lv6_fake_imc = [x for x in fake_tensors]
                    noisy_lv2_real_imc, noisy_lv4_real_imc, noisy_lv6_real_imc = [x for x in real_tensors]

                # calculate scores
                lv3_fake_score_map, lv5_fake_score_map, lv7_fake_score_map = dis(lv0_he_dev1, lv2_he_dev1, lv4_he_dev1, lv6_he_dev1, noisy_lv2_fake_imc, noisy_lv4_fake_imc, noisy_lv6_fake_imc)
                lv3_real_score_map, lv5_real_score_map, lv7_real_score_map = dis(lv0_he_dev1, lv2_he_dev1, lv4_he_dev1, lv6_he_dev1, noisy_lv2_real_imc, noisy_lv4_real_imc, noisy_lv6_real_imc)
                # retain batch dimension
                lv3_real_score_mean, lv5_real_score_mean, lv7_real_score_mean = lv3_real_score_map.mean(dim=(1, 2, 3)), lv5_real_score_map.mean(dim=(1, 2, 3)), lv7_real_score_map.mean(dim=(1, 2, 3))
                lv3_fake_score_mean, lv5_fake_score_mean, lv7_fake_score_mean = lv3_fake_score_map.mean(dim=(1, 2, 3)), lv5_fake_score_map.mean(dim=(1, 2, 3)), lv7_fake_score_map.mean(dim=(1, 2, 3))

                real_score_mean = (lv3_real_score_mean + lv5_real_score_mean + lv7_real_score_mean) / 3.0
                fake_score_mean = (lv3_fake_score_mean + lv5_fake_score_mean + lv7_fake_score_mean) / 3.0


                # LS-GAN loss, we choose a = 0 and b = c = 1 (shown to perform well and focus on generating realistic examples)
                # setting b-c = 1 and b-a = 2 gives minimization of Pearson chi2 divergence
                real_label = torch.ones(real_score_mean.size()).to(dev1)
                # don't need fake label, as it's just zeros (legacy from Simon's code)
                # fake_label = torch.zeros(fake_score_mean.size()).to(dev1)

                trans_loss = 0.5 * torch.square(fake_score_mean - real_label).mean()
                dis_loss = 0.5 * torch.square(real_score_mean - real_label).mean() + 0.5 * torch.square(fake_score_mean).mean()
                #print('Dis loss',dis_loss.item())
                dis_loss.backward()
                dis_opti.step()

                # TODO: one could decide upon updating generator based on fake_score_mean
                update_trans = update_translator_bool(rule=args.trans_update_rule, dis_fake_loss=fake_score_mean)

                ##### do a step for translator: push HE through translator to obtain fake IMC #####
                if update_trans:
                    #print('Updating translator')
                    unfreeze_params(trans)
                    # none for memory saving, see https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
                    trans_opti.zero_grad(set_to_none=True) 
                    lv2_fake_imc, lv4_fake_imc, lv6_fake_imc = trans(lv0_he_dev0, lv2_he_dev0, lv4_he_dev0, lv6_he_dev0)

                    # move generated and real imc to second GPU (for discriminator update)
                    # TODO: investigate if any downside of merging the fake_imc into a list of tensors
                    fake_tensors = [lv2_fake_imc.to(dev1), lv4_fake_imc.to(dev1), lv6_fake_imc.to(dev1)]
                    #real_tensors = [lv2_real_imc.to(dev1), lv4_real_imc.to(dev1), lv6_real_imc.to(dev1)]
                    # make IMC noisy to make things harder for discriminator
                    if args.dis_add_noise:
                        noisy_lv2_fake_imc, noisy_lv4_fake_imc, noisy_lv6_fake_imc = [add_noise(x) for x in fake_tensors]
                        #noisy_lv2_real_imc, noisy_lv4_real_imc, noisy_lv6_real_imc = [add_noise(x) for x in real_tensors]
                    else:
                        noisy_lv2_fake_imc, noisy_lv4_fake_imc, noisy_lv6_fake_imc = [x for x in fake_tensors]
                        #noisy_lv2_real_imc, noisy_lv4_real_imc, noisy_lv6_real_imc = [x for x in real_tensors]

                     # calculate scores
                    lv3_fake_score_map, lv5_fake_score_map, lv7_fake_score_map = dis(lv0_he_dev1, lv2_he_dev1, lv4_he_dev1, lv6_he_dev1, noisy_lv2_fake_imc, noisy_lv4_fake_imc, noisy_lv6_fake_imc)
                    # retain batch dimension
                    lv3_fake_score_mean, lv5_fake_score_mean, lv7_fake_score_mean = lv3_fake_score_map.mean(dim=(1, 2, 3)), lv5_fake_score_map.mean(dim=(1, 2, 3)), lv7_fake_score_map.mean(dim=(1, 2, 3))

                    # TODO: consider weighted loss (upweight lower resolution as slice-to-slice discrepancy)
                    fake_score_mean = (lv3_fake_score_mean + lv5_fake_score_mean + lv7_fake_score_mean) / 3.0

                    # define translator loss:
                    trans_loss = 0.5 * torch.square(fake_score_mean - real_label).mean()
                    trans_loss.backward()
                    trans_opti.step()

                # Note: these are training losses!!
                dis_loss_val += dis_loss.detach().cpu().item()
                trans_loss_val += trans_loss.detach().cpu().item()
                #print('Trans loss',trans_loss_val)

                x_epoch = args.save_every_x_epoch
                if i > 0 and i % (len(trainloader) // x_epoch) == 0:
                    # log loss values X times every epoch
                    dis_loss_val /= len(trainloader) / x_epoch
                    trans_loss_val /= len(trainloader) / x_epoch

                    # log for tensorboard:
                    writer.add_scalars(datetime_full_iso + "_losses",
                                       {"discriminator": dis_loss_val,
                                        "translator": trans_loss_val},
                                       global_step=epoch_idx * x_epoch + int(x_epoch * i / len(trainloader)))
                    writer.flush()

                    dis_loss_val, trans_loss_val = 0.0, 0.0


            # save models
            torch.save(trans, os.path.join(log_folder, datetime_full_iso + "_translator_" + str(epoch_idx) + ".pth"))
            torch.save(dis, os.path.join(log_folder, datetime_full_iso + "_discriminator_" + str(epoch_idx) + ".pth"))

            if dis_lr_scheduler is not None:
                dis_lr_scheduler.step()
            if trans_lr_scheduler is not None:
                trans_lr_scheduler.step()

            prof.step()
            
        # close logging writer
        writer.close()
        t1_stop = time.perf_counter()
        t2_stop = time.process_time()
        t1 = (t1_stop-t1_start)
        t2 = (t2_stop-t2_start)
        print(t2)
        memory_cost = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metainfo = pd.DataFrame({'perf_counter':t1, 'process_time': t2,
                                     'memory_cost': memory_cost}, index=[0])
        metainfo.to_csv(os.path.join(surrounding_path,'resources.tsv'), index=False, sep='\t')

    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=20))