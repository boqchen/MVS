import numpy as np
import pandas as pd
import json
import os
import argparse
import cv2
from itertools import groupby

from codebase.utils.constants import *
from codebase.utils.raw_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess and write IMC as numpy arrays.')
    parser.add_argument('--project_path', type=str, required=False, default='/raid/sonali/project_mvs/', help='path where all data, results etc for project reside')
    parser.add_argument('--input_dir', type=str, required=False, default='binary_imc_rois_raw_all', help='Folder name including input data')
    parser.add_argument('--save_dir', type=str, required=False, default='tmp', help='Folder name to save the processed data')
    parser.add_argument('--sample_list', type=str, required=False, default='', help='List of samples to parse, separated by comma (useful for a specific rerun)')
    parser.add_argument('--keep_only_mvs_rois', type=str2bool, required=False, default=True, help='Whether to write only ROIs from a pre-selected pool (MVS cohort ROIs); note: all ROIs from the input folder will be used for computation of stats and thresholds')
    parser.add_argument('--overwrite', type=str2bool, required=False, default=False, help='Whether to overwrite if output files exist')
    parser.add_argument('--arcsinh', type=str2bool, required=False, default=True)
    parser.add_argument('--clip99', type=str2bool, required=False, default=True, help='Whether to clip data to 99.99th percentile')
    parser.add_argument('--otsu_gaussian_blur_sigma', type=int, required=False, default=3, help='Sigma for the Gaussian blur used for OTSU thresholding; only used if otsu_thrs=True')
    parser.add_argument('--otsu_save_gaussian_blur', type=str2bool, required=False, default=False, help='Whether to save Gaussian blurred data; only used if otsu_thrs=True')
    parser.add_argument('--otsu_thrs', type=str2bool, required=False, default=True, help='Whether to apply OTSU thresholding (on a sample level)')
    parser.add_argument('--scale01_sample', type=str2bool, required=False, default=False, help='Whether to minmax scale within one sample data; if max is 0 then values will not be re-scaled')
    
    args = parser.parse_args()
    
    # set the paths
    PROJECT_PATH = Path(args.project_path)
    DATA_DIR = PROJECT_PATH.joinpath(DATA_DIR)
    CV_SPLIT_ROIS_PATH = PROJECT_PATH.joinpath(CV_SPLIT_ROIS_PATH)
    # make sure save_dir exists
    SAVE_PATH = DATA_DIR.joinpath(args.save_dir)
    if not os.path.exists(SAVE_PATH):
        Path(SAVE_PATH).mkdir(parents=True)
        
    existing_samples = []
    if not args.overwrite:
        existing_samples = np.unique(sorted([x.split('_')[0] for x in os.listdir(SAVE_PATH)]))

    # ------ get lists of sample_roi pairs (to process and to save)
    # get sample/ROIs list
    s_rois = [x.split('.')[0] for x in os.listdir(DATA_DIR.joinpath(args.input_dir))]
    if args.sample_list != '':
        sel_samples = args.sample_list.split(',')
        s_rois = [x for x in s_rois if x.split('_')[0] in sel_samples]
    # group rois per sample 
    def keyfunc(item):
        return item.split('_')[0]
    s_rois_grouped = [list(v) for k,v in groupby(sorted(s_rois), keyfunc)]
    print('#samples: '+str(len(s_rois_grouped)))
    # get filtered sample/ROIs list
    if args.keep_only_mvs_rois:
        cv = json.load(open(CV_SPLIT_ROIS_PATH))
        keep_rois = cv['split0']['train']
        keep_rois.extend(cv['split0']['valid'])
        keep_rois.extend(cv['split0']['test'])
       
    
    # ----- Loop through all the samples, perform transformations and save ROIs -----   
    for j,sample_rois in enumerate(s_rois_grouped):
        print(j)
        if sample_rois[0].split('_')[0] in existing_samples:
            continue
        # load all ROIs for a given sample
        imc_np_sample = []
        for sample_roi in sample_rois:
            roi = np.load(DATA_DIR.joinpath(args.input_dir,sample_roi+'.npy'))
            # make sure all ROIs are 1000x1000
            assert np.isclose(roi.shape[0],1000) and np.isclose(roi.shape[1], 1000), 'The ROI '+sampe_roi+' has incorrect shape, aborting' 
            imc_np_sample.extend(roi)
        imc_np_sample = np.array(imc_np_sample)

        if args.clip99:
            # TODO: double-check that correct
            q99 = np.percentile(imc_np_sample, 99.99, axis=[0,1])
            imc_np_sample = np.clip(imc_np_sample, a_min=0, a_max=q99)

        if args.arcsinh:
            def trafo(x): return np.log(x + np.sqrt(x**2 + 1))
            imc_np_sample = trafo(imc_np_sample)
        
        if args.otsu_thrs:
            imc_np_sample, _ = apply_otsu_thresholding(imc_np_sample, args.otsu_gaussian_blur_sigma, return_blurred=args.otsu_save_gaussian_blur)
            
        if args.scale01_sample:
            def min_max_scale_nonzero(x):
                x_min = np.min(x)
                x_max = np.max(x)
                if np.isclose(x_max,0):
                    return x
                else:
                    return (x-x_min)/(x_max - x_min)
            imc_np_sample = np.apply_along_axis(min_max_scale_nonzero, 2, imc_np_sample)
            
        # split sample-level data into ROIs and save
        for i,sample_roi  in enumerate(sample_rois):
            if args.keep_only_mvs_rois:
                if sample_roi not in keep_rois:
                    continue
            imc_np_sample_roi = imc_np_sample[(i*1000):((i+1)*1000),:,:]
            np.save(SAVE_PATH.joinpath(sample_roi+'.npy'), imc_np_sample[(i*1000):((i+1)*1000),:,:])
