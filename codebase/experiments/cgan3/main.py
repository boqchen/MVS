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
from fractions import Fraction
from tqdm import tqdm
from distutils.util import strtobool
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import torchvision.transforms.functional as ttf
from torch.optim import lr_scheduler
from torch.nn import init

from torchinfo import summary

from codebase.utils.constants import *
from codebase.utils.raw_utils import *
from codebase.utils.zipper import do_zip
from codebase.experiments.cgan3.training_helpers import *
from codebase.experiments.cgan3.network import * #Translator, Discriminator
from codebase.experiments.cgan3.loaders import CGANDataset_v2
from codebase.utils.eval_utils import get_protein_list
from codebase.utils.tb_logger import divide_into_patches, stitch_patches, colorize_images, TBLogger
from codebase.utils.inference_utils import get_tensor_from_numpy, get_target_shapes, pad_img
from codebase.utils.loss_utils import *
from codebase.utils.gauss_pyramid import Gauss_Pyramid_Conv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cGAN.')
    parser.add_argument('--no_zip', action='store_true', help='By default, the script saves the state of the code base as a timestamped zip file. Setting this flag turns it off.')
    parser.add_argument('--seed', type=int, required=False, default=0, help='Random seed')
    parser.add_argument('--n_step', type=int, required=False, default=360000, help='Number of steps to train')
    parser.add_argument('--cv_split', type=str, required=False, default=None, help='Selected CV split, if None then the splitting used for the report is used')
    parser.add_argument('--protein_set', type=str, required=False, default='full', help='which protein set to use')
    parser.add_argument('--restore_model', type=str, required=False, default="NO_FILE", help='Model checkpoint dictionary to restore the model from')
    parser.add_argument('--dis_lr_scheduler', type=str, required=False, default='fixed', help='Learning rate scheduler for discriminator network {step, plateau, cosine}')
    parser.add_argument('--trans_lr_scheduler', type=str, required=False, default='fixed', help='Learning rate scheduler for translator network {step, plateau, cosine}')
    parser.add_argument('--trans_update_rule', type=str, required=False, default='prob', help='Rule for updating translator {always, prob, dis_loss}')
    parser.add_argument('--imc_prep_seq', type=str, required=False, default='raw_median_arc', help='Sequence of data preprocessing steps (needed to select form which folder to load the data)')
    parser.add_argument('--trans_weights_init', type=str, required=False, default='xavier', help='Weights initialization for translator')
    parser.add_argument('--dis_weights_init', type=str, required=False, default='xavier', help='Weights initialization for discriminator')
    parser.add_argument('--blur_gt', type=str2bool, required=False, default=False, help='Whether to blur ground-truth IMC data')
    parser.add_argument('--dis_add_noise', type=str, required=False, default='False_0.0', help='Whether to add noise for training discriminator. Pass as string: True/False followed by probability of adding noise. Eg "True_0.5"')
    parser.add_argument('--submission_id', type=str, required=True, help='Job submission ID')
    parser.add_argument('--standardize_imc', type=str2bool, required=False, default=True, help='Whether to standardize IMC data using cohort mean and stddev per channel')
    parser.add_argument('--scale01_imc', type=str2bool, required=False, default=True, help='Whether to scale IMC data batch-wise to [0,1] interval per channel')
    parser.add_argument('--patch_size', type=int, required=False, default=256, help='Patch size')
    parser.add_argument('--batch_size', type=int, required=False, default=16, help='Batch size, works with single gpu for batch size upto 32')
    parser.add_argument('--save_every_x_epoch', type=int, required=False, default=20, help='Save logs every 1/x epoch.')
    parser.add_argument('--save_chkpt_every_x_epoch', type=int, required=False, default=2, help='Save model checkpoint every 1/x epoch.')
    parser.add_argument('--n_gpus', type=int, required=False, default=1, help='Number of gpus required for training. Acceptable values: {1,2}')
    parser.add_argument('--model_depth', type=int, required=False, default=6, help='depth of the translator and discriminator model, tested for depth 5 to 8')
    parser.add_argument('--last_activation', type=str, required=False, default='relu', help='activation for the last layers in generator model, eg relu/identity..')
    parser.add_argument('--multiscale', type=str2bool, required=False, default=True, help='IMC to be generated at single or multiple scales')
    parser.add_argument('--asymmetric', type=str2bool, required=False, default=True, help='Asymmertric or symmetric u-net generator, if IMC and H&E size same then symmetric setting')
    parser.add_argument('--debug_set', type=str2bool, required=False, default=False, help='Use only a small subset of data for debugging purposes')
    parser.add_argument('--data_path', type=str, required=False, default='/cluster/work/grlab/projects/projects2021-multivstain/data/tupro/', help='path where paired H&E and IMC data resides')
    parser.add_argument('--factor_len_dataloader', type=float, required=False, default='8.0', help='factor for lenght of data loader')
    parser.add_argument('--weight_multiscale', type=str, required=False, default='1/3,1/3,1/3', help='weight of imc downsamples in multiscale realness score TODO: add in which order the weights ')
    parser.add_argument('--which_cluster', type=str, required=True, help='biomed or dgx:gpu')
    parser.add_argument('--project_path', type=str, required=False, default='/raid/sonali/project_mvs/', help='path where all data, results etc for project reside')
    parser.add_argument('--which_HE', type=str, required=False, default='new', help='old or new HE scanned regions for model training')
    parser.add_argument('--weight_L1', type=float, required=False, default=0.0, help='weight for L1 loss in translator loss term')
    parser.add_argument('--p_flip_jitter_hed_affine', type=str, required=False, default='0.5,0.0,0.5,0.5', help='probability for different augmentations. Order: flip/rot, he color jitter, hed color aug, he affine')
    parser.add_argument('--which_translator', type=str, required=False, default='old', help='options: "no_checkerboard", "old", "classic_unet", "classic_unet_no_checkerboard"')
    parser.add_argument('--finetune_samples', type=str, required=False, default=None, help='name of samples for which we want to fine-tune the already trained model') # 'MADUBIP,MAHEFOG,MEGEGUJ'
    parser.add_argument('--use_roi_weights', type=str2bool, required=False, default=False, help='If True then the roi weights calculated based on marker sparsity are used during roi sampling in dataloader') 
    parser.add_argument('--weight_r1', type=float, required=False, default=1., help='weight of r1 loss')
    parser.add_argument('--r1_gamma', type=float, required=False, default=.0002, help='gamma of r1 loss')
    parser.add_argument('--r1_interval', type=int, required=False, default=16, help='interval of calculating r1 loss')
    parser.add_argument('--log_interval', type=int, required=False, default=10, help='interval of logging loss and lr')
    parser.add_argument('--vis_interval', type=int, required=False, default=5000, help='interval of visualizing val samples')
    parser.add_argument('--n_vis_samples', type=int, required=False, default=4, help='num of visualizing val samples')
    parser.add_argument('--save_interval', type=int, required=False, default=5000, help='interval of saving state dict')
    parser.add_argument('--dis_mbdis', type=bool, required=False, default=True, help='whether to enable mini-batch discrimination')
    parser.add_argument('--eq_lr', type=bool, required=False, default=False, help='whether to use equalised learning rate layers')
    parser.add_argument('--weight_ASP', type=float, required=False, default=0., help='weight of adaptive supervised patchNCE loss')
    parser.add_argument('--enable_RLW', type=str2bool, required=False, default=False, help='If true then use random task weighting')
    parser.add_argument('--multiscale_L1', type=str2bool, required=False, default=False, help='If true then use loss as described in BCI paper')
    parser.add_argument('--weight_multiscale_L1', type=float, required=False, default=1., help='weight given to pyramid/GP loss')

    args = parser.parse_args()
    
    t1_start = time.perf_counter()
    t2_start = time.process_time()

    # set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # setting device 
    if 'dgx' in args.which_cluster:
        # here
        os.environ['CUDA_VISIBLE_DEVICES'] = args.which_cluster.split(':')[1] 
        dev0 = torch.device('cuda')

        # to here

        #dev0 = torch.device('cuda:' + args.which_cluster.split(':')[1])
        args.data_path = os.path.join(args.project_path, 'data/tupro')

    else:
#         dev0 = torch.device("cpu")
        dev0 = torch.device('cuda:0')
    
    # save path for logs, experiment results ...
    save_path = os.path.join(args.project_path, 'results', args.submission_id)
    surrounding_path = save_path
    print(f"Saving exps at:{surrounding_path}")

    if args.n_gpus==1:  
        dev1 = dev0
    elif args.n_gpus==2: 
        dev1 = torch.device('cuda:1')
    
    print(args.which_cluster, dev0)

    # get current datetime up to seconds to create log folder
    datetime_full_iso = get_datetime()
    log_folder = os.path.join(surrounding_path, "tb_logs")
    print(f"Saving logs at:{log_folder}")
    if not os.path.exists(log_folder):
        Path(log_folder).mkdir(parents=True)
    tb_logger = TBLogger(log_dir=log_folder)

    if not args.no_zip:
        zip_folder = os.path.join(surrounding_path, "code_zips")
        if not os.path.exists(zip_folder): 
            Path(zip_folder).mkdir(parents=True)
        do_zip(label=datetime_full_iso, save_folder=zip_folder)
        
    # save argument values into a txt file
    with open(os.path.join(surrounding_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print('Created output folders')
    
    # log parameters to tensorboard
    for arg_name, arg_val in vars(args).items():
        tb_logger.run(func_name="add_text", tag=arg_name, text_string=str(arg_val))
    
    protein_subset = get_protein_list(args.protein_set)
    
    # getting samples for desied align_set
    if args.finetune_samples==None: 
        train_aligns = get_aligns(args.project_path, cv_split=args.cv_split, protein_set=args.protein_set, aligns_set='train')
    else: 
        samples=args.finetune_samples.split(',')
        finetune_aligns= get_aligns(args.project_path,cv_split=args.cv_split,protein_set=args.protein_set,aligns_set='train_finetune')
        train_aligns = []
        for align in finetune_aligns:
            if align['sample'] in samples:
                train_aligns.append(align)
    
    if args.debug_set:
        train_aligns = train_aligns[:1]
    batch_size = args.batch_size
    # TODO: atm only patch size of 256 works!!
    patch_size = args.patch_size #256 µms, should be a multiple of 16
    assert patch_size==256, 'Atm patch size of 256 is the only implemented option.'
    print('Read all args.')
    
    # the size of the different IMC downsamples when multiscale setting
    if args.multiscale: 
        imc_sizes = [] # eg: [patch_size//16, patch_size//4]
        for j in reversed(range(args.model_depth//2 -1)):
            imc_sizes.append(patch_size//(2**(2*(j+1))))
        
        weight_downsamples = [float(Fraction(x)) for x in args.weight_multiscale.split(',')]

    print('Train set with '+str(len(train_aligns))+' ROIs')

        
    p_flip_jitter_hed_affine = list(map(float, args.p_flip_jitter_hed_affine.split(',')))

    train_ds = CGANDataset_v2(args.project_path, align_results=train_aligns,
                        name="Train",
                        data_path=args.data_path,
                        patch_size=patch_size,
                        protein_subset=protein_subset,
                        imc_prep_seq=args.imc_prep_seq,
                        cv_split=args.cv_split,
                        standardize_imc=args.standardize_imc,
                        scale01_imc=args.scale01_imc,
                        factor_len_dataloader=args.factor_len_dataloader, 
                        which_HE=args.which_HE, 
                        p_flip_jitter_hed_affine=p_flip_jitter_hed_affine, 
                        use_roi_weights = args.use_roi_weights)

    print('Loaded data')
    trainloader = DataLoader(train_ds,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=8, 
                             drop_last=True)
    trainloader_iter = enumerate(trainloader)
        
    print('HE_ROI_STORAGE: ', train_ds.HE_ROI_STORAGE)
    print('IMC_ROI_STORAGE: ', train_ds.IMC_ROI_STORAGE)
    
    # val images
    val_tsfm = train_ds.imc_transforms_val
    val_rois = json.load(open(os.path.join(args.project_path, CV_SPLIT_ROIS_PATH)))[args.cv_split]['valid']
    val_rois = random.sample(val_rois, min(args.n_vis_samples, len(val_rois)))
    
    val_he_roi_tensors_lvl0 = []
    val_imc_roi_tensors_lvl0 = []

    for roi in val_rois:
        val_he_roi_np_lvl0 = np.load(os.path.join(train_ds.HE_ROI_STORAGE, roi + ".npy"), mmap_mode='r')
        val_he_roi_tensor_lvl0 = get_tensor_from_numpy(val_he_roi_np_lvl0).to(dev0)
        imc_desired_shapes = get_target_shapes(args.model_depth, val_he_roi_np_lvl0.shape[0])
        val_he_roi_tensor_lvl0 = pad_img(val_he_roi_tensor_lvl0, val_he_roi_np_lvl0.shape[0]).to(dev0)
        
        val_he_roi_tensors_lvl0.append(val_he_roi_tensor_lvl0)
    
        val_imc_roi_np_lvl0 = np.load(os.path.join(train_ds.IMC_ROI_STORAGE, roi + ".npy"), mmap_mode='r')
        val_imc_roi_np_lvl0 = val_imc_roi_np_lvl0[:, :, train_ds.channel_list]
        val_imc_roi_tensor_lvl0 = get_tensor_from_numpy(val_imc_roi_np_lvl0)
        
        val_imc_roi_tensors_lvl0.append(val_imc_roi_tensor_lvl0)
        
    # for i, val_he_roi_tensor_lvl0 in enumerate(val_he_roi_tensors_lvl0):
    #     # val_he_roi_tensor_lvl0 = ttf.center_crop(val_he_roi_tensor_lvl0, 1024)
    #     he = torch.clip(val_he_roi_tensor_lvl0, 0., 1.).squeeze(dim=0)
    #     tb_logger.run(func_name="add_image", tag=f"{val_rois[i]}_H&E", img_tensor=he, global_step=0)
        
    # log ground truth IMC images
    for i, val_imc_roi_tensor_lvl0 in enumerate(val_imc_roi_tensors_lvl0):
        # val_imc_roi_tensor_lvl0 = ttf.center_crop(val_imc_roi_tensor_lvl0, 256)
        # imc_real = torch.clip(val_imc_roi_tensor_lvl0, 0., 1.)
        imc_real = val_imc_roi_tensor_lvl0
        imc_real = F.interpolate(imc_real, size=(100, 100), mode='bilinear', align_corners=False)
        imc_real = imc_real.permute(1, 0, 2, 3) # channel --> batch
        imc_real = colorize_images(imc_real)
        imc_real = make_grid(imc_real, 4, 3)
        tb_logger.run(func_name="add_image", tag=f"{val_rois[i]}_IMC_real", img_tensor=imc_real, global_step=0)

    # create models
    if args.which_translator=="no_checkerboard":
        trans = unet_translator(n_output_channels=len(protein_subset), depth=args.model_depth, 
                                flag_asymmetric=args.asymmetric, flag_multiscale=args.multiscale,
                                last_activation=args.last_activation,which_decoder='conv', encoder_padding=1, decoder_padding=1, eq_lr=args.eq_lr)   

    elif args.which_translator=="old": 
        trans = unet_translator(n_output_channels=len(protein_subset), depth=args.model_depth, 
                                flag_asymmetric=args.asymmetric, flag_multiscale=args.multiscale,
                                last_activation=args.last_activation,which_decoder='convT', encoder_padding=1, decoder_padding=1, eq_lr=args.eq_lr) 
        
    elif args.which_translator=="classic_unet": 
        trans = unet_translator(n_output_channels=len(protein_subset), depth=args.model_depth, 
                                flag_asymmetric=args.asymmetric, flag_multiscale=args.multiscale,
                                last_activation=args.last_activation,which_decoder='convT', encoder_padding=0, decoder_padding=0, eq_lr=args.eq_lr) 
        
    elif args.which_translator=="classic_unet_no_checkerboard": 
        trans = unet_translator(n_output_channels=len(protein_subset), depth=args.model_depth, 
                                flag_asymmetric=args.asymmetric, flag_multiscale=args.multiscale,
                                last_activation=args.last_activation,which_decoder='conv', encoder_padding=0, decoder_padding=1, eq_lr=args.eq_lr) 
    print(trans)
    # create ema model for translater 
    trans_ema = deepcopy(trans)
    trans_ema.to(dev0)
    freeze_params(trans_ema)
    trans_ema.eval()

    trans_opti = get_optimizer(trans, 'translator', args.trans_lr_scheduler)
    
    
    dis = Discriminator(n_output_channels=len(protein_subset), depth=args.model_depth, 
                flag_asymmetric=args.asymmetric, flag_multiscale=args.multiscale, mbdis=args.dis_mbdis, eq_lr=args.eq_lr)
    print(dis)
    if args.weight_r1 > 0:
        lazy_c = args.r1_interval / (args.r1_interval + 1) if args.r1_interval > 1 else 1
        dis_opti = get_optimizer(dis, 'discriminator', args.dis_lr_scheduler, lazy_c)
    else:
        dis_opti = get_optimizer(dis, 'discriminator', args.dis_lr_scheduler)
        
    # L1 loss
    L1loss = torch.nn.L1Loss(reduction='none')
    
    # contrastive loss
    netF = None
    netF_opti = None
    if not np.isclose(args.weight_ASP, 0.0): 
        # initialise vgg19
        MODELDIR = os.path.join(args.project_path, PRETRAINED_DIR)
        vgg = VGG19(MODELDIR).to(dev0)
        freeze_params(vgg)
        vgg.eval()
             
        # initialise netF
        netF = PatchSampleF().to(dev0)
        dummy_feats = vgg(torch.zeros([args.batch_size, 1, 256, 256], device=dev0)) # use dummy data for netF initialisation
        _ = netF(dummy_feats, 256, None)
        patchNCELoss = PatchNCELoss(batch_size=args.batch_size, total_step=args.n_step, n_step_decay=10000).to(dev0)
        netF_opti = get_optimizer(netF, 'netF', args.dis_lr_scheduler)
            
    if args.restore_model == "NO_FILE":
        if args.trans_weights_init != 'no_init':
            init_weights(trans, init_type=args.trans_weights_init, init_gain=0.02)
        trans_lr_scheduler = get_lr_scheduler(trans_opti, args.trans_lr_scheduler)
        if args.dis_weights_init != 'no_init':
            init_weights(dis, init_type=args.dis_weights_init, init_gain=0.02)
        dis_lr_scheduler = get_lr_scheduler(dis_opti, args.dis_lr_scheduler)
        epoch_number = 0
    else:
        # restore model from checkpoint
        print('Restoring from a checkpoint.')
        checkpoint = torch.load(args.restore_model, map_location=dev0)
        trans.load_state_dict(checkpoint['trans_state_dict'])
        trans_opti.load_state_dict(checkpoint['trans_optimizer_state_dict'])
        optimizer_to(trans_opti, dev0)
        trans_lr_scheduler = checkpoint['trans_lr_scheduler']
        dis.load_state_dict(checkpoint['dis_state_dict'])
        dis_opti.load_state_dict(checkpoint['dis_optimizer_state_dict'])
        optimizer_to(dis_opti, dev1)
        dis_lr_scheduler = checkpoint['dis_lr_scheduler']
        epoch_number = checkpoint['epoch']+1
        print('Model restored from epoch number: ', epoch_number)
        
    trans.to(dev0)
    dis.to(dev1)
    
    if args.blur_gt:
        spatial_denoise = torchvision.transforms.GaussianBlur(3, sigma=1)

    # if add noise in discriminator 
    dis_add_noise = strtobool(args.dis_add_noise.split('_')[0])
    p_dis_add_noise = float(args.dis_add_noise.split('_')[1])

    unfreeze_params(dis)
    unfreeze_params(trans)
    trans.train()
    dis.train()
    
    step = 0 # global step
    d_step = 0 # dis step
    g_step = 0 # gen step
    
    if args.multiscale_L1: 
        # pyr_conv = Gauss_Pyramid_Conv(num_high=5)
        # gp_weights = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 1.0]
        pyr_conv = Gauss_Pyramid_Conv(num_high=3)
        gp_weights = [0.0625, 0.125, 0.25, 1.0]

    print('Initalization finished')

#     for _ in tqdm(range(args.n_step)):
    for _ in (range(args.n_step)):

        try:
            _, batch = trainloader_iter.__next__()
        except:
            trainloader_iter = enumerate(trainloader)
            _, batch = trainloader_iter.__next__()
               
        lv2_real_imc = batch["imc_patch"].to(dev0)
        lv0_he_dev0 = batch["he_patch"].to(dev0)
        
        ###########################################
        ########### updata discriminator ##########
        ###########################################
        
        # to avoid mixing gradients across minibatches (.backward() accumulates gradients!)
        # none for memory saving, see https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        dis_opti.zero_grad(set_to_none=True)
        freeze_params(trans)
        with torch.no_grad():
            fake_imcs = trans(lv0_he_dev0)

        # prepare different resolutions of IMC and HE
        real_imcs = []
        with torch.no_grad():
            if args.blur_gt:
                lv2_real_imc = spatial_denoise(lv2_real_imc)
            if args.multiscale:
                real_imcs.extend(resize_tensor(lv2_real_imc, imc_sizes))
            real_imcs.append(lv2_real_imc)
                    
        if dis_add_noise:
            # make IMC noisy to make things harder for discriminator
            fake_imcs = [add_noise_prob(x, p_dis_add_noise) for x in fake_imcs]
            real_imcs = [add_noise_prob(x, p_dis_add_noise) for x in real_imcs]
            
        # calculate scores 
        real_imcs = [real_imc.to(dev1) for real_imc in real_imcs]
        if args.n_gpus==2:
            lv0_he_dev1 = batch["he_patch"].to(dev1)
            fake_imcs = [fake_imc.to(dev1) for fake_imc in fake_imcs]                
            fake_score_maps = dis(lv0_he_dev1, fake_imcs)            
            real_score_maps = dis(lv0_he_dev1, real_imcs)
        else:
            fake_score_maps = dis(lv0_he_dev0, fake_imcs)            
            real_score_maps = dis(lv0_he_dev0, real_imcs) 

        # retain batch dimension
        real_score_means = [real_score_map.mean(dim=(1, 2, 3))for real_score_map in real_score_maps]
        fake_score_means = [fake_score_map.mean(dim=(1, 2, 3))for fake_score_map in fake_score_maps]
        
        real_score_mean = sum(real_score_means)/len(real_score_means)
        fake_score_mean = sum(fake_score_means)/len(fake_score_means)
                    
        # LS-GAN loss, we choose a = 0 and b = c = 1 (shown to perform well and focus on generating realistic examples)
        # setting b-c = 1 and b-a = 2 gives minimization of Pearson chi2 divergence
        real_label = torch.ones(real_score_mean.size()).to(dev1)
        # don't need fake label, as it's just zeros (legacy from Simon's code)
        # fake_label = torch.zeros(fake_score_mean.size()).to(dev1)
            
        # r1 penalty loss
        if (args.weight_r1 > 0) & (step % args.r1_interval == 0):
            if args.n_gpus==2:
                r1_loss = get_r1(dis, lv0_he_dev1, real_imcs, gamma_0=args.r1_gamma, lazy_c=lazy_c)
            else:
                r1_loss = get_r1(dis, lv0_he_dev0, real_imcs, gamma_0=args.r1_gamma, lazy_c=lazy_c)
            tb_logger.run(func_name="log_scalars", metric_dict={"r1_loss": r1_loss.item()}, step=step)
        else:
            r1_loss = 0.0
              
        dis_loss = 0.5 * torch.square(real_score_mean - real_label).mean() + 0.5 * torch.square(fake_score_mean).mean()
        total_dis_loss = dis_loss + args.weight_r1 * r1_loss
        
        total_dis_loss.backward()
        dis_opti.step()
        
        ###########################################
        ############# updata translator ###########
        ###########################################
        
        # Boolean whether to update translator in this step
        update_trans = update_translator_bool(rule=args.trans_update_rule, dis_fake_loss=fake_score_mean)
    
        if update_trans:
            unfreeze_params(trans)
            # none for memory saving, see https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
            trans_opti.zero_grad(set_to_none=True)
            if netF_opti is not None:
                netF_opti.zero_grad(set_to_none=True)
            fake_imcs = trans(lv0_he_dev0)
                            
            # make IMC noisy to make things harder for discriminator
            if dis_add_noise:
                fake_imcs = [add_noise_prob(x, p_dis_add_noise) for x in fake_imcs]

                # calculate scores
            if args.n_gpus==2:
                fake_imcs = [fake_tensor.to(dev1) for fake_tensor in fake_imcs]
                fake_score_maps = dis(lv0_he_dev1, fake_imcs)
            else:
                fake_score_maps = dis(lv0_he_dev0, fake_imcs)

            # retain batch dimension
            fake_score_means = [fake_score_map.mean(dim=(1, 2, 3)) for fake_score_map in fake_score_maps]
            fake_score_mean = sum(fake_score_means)/len(fake_score_means)            

            # define translator loss:
            trans_loss = 0.
            gen_loss = 0.5 * torch.square(fake_score_mean - real_label).mean()
            trans_loss += gen_loss
            
            n_channel = real_imcs[-1].shape[1] # num of markers/tasks
            batch_weight = F.softmax(torch.randn(n_channel), dim=-1).to(dev0) if args.enable_RLW else torch.ones(n_channel).to(dev0)
            
            if not np.isclose(args.weight_L1, 0.0): 
                if args.multiscale_L1: 
                    loss_pyramid = [L1loss(pf, pr).mean(dim=(2, 3)) for pf, pr in zip(pyr_conv(fake_imcs[-1]), pyr_conv(real_imcs[-1]))]
                    loss_pyramid = [l * w for l, w in zip(loss_pyramid, gp_weights)]
                    loss_pyramid = [torch.mul(l, batch_weight).mean() for l in loss_pyramid]
                    loss_GP = torch.mean(torch.stack(loss_pyramid))
                    loss_GP *= args.weight_multiscale_L1
                    if (step % args.log_interval == 0) & (step > 0):
                        tb_logger.run(func_name="log_scalars", metric_dict={"GP_loss": loss_GP.item()}, step=step)
                    trans_loss += 0.5 * loss_GP
                else:
                    loss_l1 = L1loss(fake_imcs[-1], real_imcs[-1]).mean(dim=(2, 3))
                    loss_l1 = torch.mul(loss_l1, batch_weight).mean()
                    loss_l1 *= args.weight_L1 
                    trans_loss += 0.5 * loss_l1
                    if (step % args.log_interval == 0) & (step > 0):
                        tb_logger.run(func_name="log_scalars", metric_dict={"L1_loss": loss_l1.item()}, step=step)
                        
            if not np.isclose(args.weight_ASP, 0.0):
                total_asp_loss = []
                
                # calculate ASP loss between real_B and fake_B
                for i in range(n_channel):
                    feat_real_B = vgg(real_imcs[-1][:, i:i+1, :, :])
                    feat_fake_B = vgg(fake_imcs[-1][:, i:i+1, :, :])
                    n_layers = len(feat_fake_B)
                    feat_q = feat_fake_B
                    feat_k = feat_real_B
                    feat_k_pool, sample_ids = netF(feat_k, 256, None)
                    feat_q_pool, _ = netF(feat_q, 256, sample_ids)
                    
                    total_asp_loss_per_channel = 0.0
                    for f_q, f_k in zip(feat_q_pool, feat_k_pool):
                        loss = patchNCELoss(f_q, f_k, step)
                        total_asp_loss_per_channel += loss.mean()
                    total_asp_loss_per_channel /= n_layers
                    total_asp_loss.append(total_asp_loss_per_channel)
                
                total_asp_loss = torch.tensor(total_asp_loss, device=dev0)
                loss_asp = torch.mul(total_asp_loss, batch_weight).mean()
                loss_asp *= args.weight_ASP
                if (step % args.log_interval == 0) & (step > 0):
                    tb_logger.run(func_name="log_scalars", metric_dict={"ASP_loss": loss_asp.item()}, step=step)
                trans_loss += 0.5 * loss_asp
                        
            trans_loss.backward()
            trans_opti.step()
            
            if netF_opti is not None:
                netF_opti.step()
         
            # update trans_ema
            with torch.no_grad():
                decay=0.9999 if step >= 5000 else 0 
                # lin. interpolate and update
                for p_ema, p in zip(trans_ema.parameters(), trans.parameters()):
                    p_ema.copy_(p.lerp(p_ema, decay))
                # copy buffers
                for (b_ema_name, b_ema), (b_name, b) in zip(trans_ema.named_buffers(), trans.named_buffers()):
                    if "num_batches_tracked" in b_ema_name:
                        b_ema.copy_(b)
                    else:
                        b_ema.copy_(b.lerp(b_ema, decay))
                    
            g_step += 1
            
        # Record loss values every 10 steps
        if (step % args.log_interval == 0) & (step > 0):
            loss_dict = {
                "loss_dis": dis_loss.item(),
                "loss_trans": gen_loss.item(),
            }
            tb_logger.run(func_name="log_scalars", metric_dict=loss_dict, step=step)
            
            lr_dict = {
                "lr_dis": dis_opti.param_groups[0]['lr'],
                "lr_trans": trans_opti.param_groups[0]['lr']
            }
            tb_logger.run(func_name="log_scalars", metric_dict=lr_dict, step=step)
        
        if dis_lr_scheduler is not None:
            dis_lr_scheduler.step()
        if trans_lr_scheduler is not None:
            trans_lr_scheduler.step()
    
        # log ema samples every 1k steps
        if (step % args.vis_interval == 0) & (step != 0):
            # set to eval mode
            trans.eval()
            with torch.no_grad():
                for i, val_he_roi_tensor_lvl0 in enumerate(val_he_roi_tensors_lvl0):
                    # patches = divide_into_patches(val_he_roi_tensor_lvl0)
                    # curr_translated_patches = []
                    # ema_translated_patches = []
                    # for patch in patches:
                    #     curr_translated_patch = trans(patch)[-1]
                    #     ema_translated_patch = trans_ema(patch)[-1]
                    #     curr_translated_patches.append(curr_translated_patch)
                    #     ema_translated_patches.append(ema_translated_patch)
                        
                    # imc_pred_curr = stitch_patches(curr_translated_patches)
                    # imc_pred_ema = stitch_patches(ema_translated_patches)
                     
                    imc_pred_curr = trans(val_he_roi_tensor_lvl0)[-1] # only visualize the final output 
                    imc_pred_curr = torch.clip(imc_pred_curr, 0., 1.)
                    imc_pred_curr = F.interpolate(imc_pred_curr, size=(100, 100), mode='bilinear', align_corners=False)
                    imc_pred_curr = imc_pred_curr.permute(1, 0, 2, 3) # channel --> batch
                    imc_pred_curr = colorize_images(imc_pred_curr)
                    imc_pred_curr = make_grid(imc_pred_curr, 4, 3)
                    tb_logger.run(func_name="add_image", tag=f"{val_rois[i]}_IMC_pred_curr", img_tensor=imc_pred_curr, global_step=step)
                
                    # IMC predicted by ema smoothed generator
                    imc_pred_ema = trans_ema(val_he_roi_tensor_lvl0)[-1] # only visualize the final output 
                    imc_pred_ema = torch.clip(imc_pred_ema, 0., 1.)
                    imc_pred_ema = imc_pred_ema.permute(1, 0, 2, 3) # channel --> batch
                    imc_pred_ema = F.interpolate(imc_pred_ema, size=(100, 100), mode='bilinear', align_corners=False)
                    imc_pred_ema = colorize_images(imc_pred_ema)
                    imc_pred_ema = make_grid(imc_pred_ema, 4, 3)
                    tb_logger.run(func_name="add_image", tag=f"{val_rois[i]}_IMC_pred_ema", img_tensor=imc_pred_ema, global_step=step)
                    
        # set back to train mode
        trans.train()
           
        # save checkpoint as dictionary every 5k steps
        if (step % args.save_interval == 0) & (step != 0):
            torch.save({'step': step,
                        'trans_state_dict':trans.state_dict(),
                        'trans_optimizer_state_dict':trans_opti.state_dict(),
                        'trans_lr_scheduler':trans_lr_scheduler,
                        'trans_ema_state_dict':trans_ema.state_dict(),
                        'dis_state_dict':dis.state_dict(),
                        'dis_optimizer_state_dict':dis_opti.state_dict(),
                        'dis_lr_scheduler':dis_lr_scheduler,},
                        os.path.join(log_folder, f"{int(step/1000)}K_model_state.pt"))
            
            torch.save(trans_ema, os.path.join(log_folder, f"{int(step/1000)}K_translator.pth"))
            print('step: ', int(step/1000), 'K')

            
        d_step += 1
        step += 1
            
    # close logging writer
    tb_logger.close()
    t1_stop = time.perf_counter()
    t2_stop = time.process_time()
    t1 = (t1_stop-t1_start)
    t2 = (t2_stop-t2_start)
    print(t2)
    memory_cost = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metainfo = pd.DataFrame({'perf_counter':t1, 'process_time': t2,
                                 'memory_cost': memory_cost}, index=[0])
    metainfo.to_csv(os.path.join(surrounding_path,'resources.tsv'), index=False, sep='\t')
