import numpy as np
import pandas as pd
from skimage import draw
from pathlib import Path
import argparse
import os
import json

from codebase.utils.constants import *
from codebase.utils.raw_utils import str2bool

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess and write IMC as numpy arrays.')
    parser.add_argument('--project_path', type=str, required=False, default='/raid/sonali/project_mvs/', help='path where all data, results etc for project reside')
    parser.add_argument('--input_path', type=str, required=False, default='data/tupro/binary_imc_rois_raw/', help='Path to load raw data from just to get the right shape of the arrays')
    parser.add_argument('--sample_list', type=str, required=False, default='', help='List of samples to parse, separated by comma')
    parser.add_argument('--overwrite', type=str2bool, required=False, default=False, help='Whether to overwrite if output files exist')
    parser.add_argument('--radius', type=int, required=False, default=5, help='Radius of the circle to be drawn around the XY cell coordinates')
    parser.add_argument('--cts_col', type=str, required=False, default='ct_basic', help='Name of column with cell-type information')
    
    args = parser.parse_args()

    PROJECT_PATH = Path(args.project_path)
    INPUT_PATH = PROJECT_PATH.joinpath(args.input_path)
    save_fdir = '-'.join(['imc','celltype_masks',args.cts_col,'r'+str(args.radius)])
    SAVE_PATH = PROJECT_PATH.joinpath('meta',save_fdir)
    if not os.path.exists(SAVE_PATH): 
        os.makedirs(SAVE_PATH)
    
    cts = pd.read_csv(PROJECT_PATH.joinpath('data/tupro/imc_updated/coldata.tsv'), sep='\t')
    cts_uq = sorted(cts[args.cts_col].unique())

    
    if args.sample_list != '':
        sample_rois = args.sample_list.split(',')
    else:
        # load from cv splits
        cv = json.load(open(PROJECT_PATH.joinpath(CV_SPLIT_ROIS_PATH)))
        sample_rois = cv['split3']['train']
        sample_rois.extend(cv['split3']['valid'])
        sample_rois.extend(cv['split3']['test'])


    for s_roi in sample_rois:
        if not args.overwrite:
            if os.path.exists(SAVE_PATH.joinpath(s_roi+'.npy')):
                next
            
        x_max, y_max, _ = np.load(INPUT_PATH.joinpath(s_roi+'.npy'), mmap_mode='r').shape
        df = cts.loc[cts.sample_roi==s_roi,:]
        
        arr = np.zeros((x_max, y_max, len(cts_uq)))
        for i in range(df.shape[0]):
            object_df = df.iloc[i,:]
            # the coordinates of the ROIs are flipped wrt cts
            x0 = object_df['Y']
            y0 = object_df['X']
            cts_idx = [j for j,x in enumerate(cts_uq) if x==object_df[args.cts_col]]
            rr, cc = draw.circle_perimeter(x0, y0, radius=args.radius, shape=arr.shape, method='andres')
            # circle contour
            arr[rr, cc, cts_idx] = 1
            # fill the circle
            for x in range(abs(x0-args.radius),(x0+args.radius)):
                for y in range(abs(y0-args.radius),(y0+args.radius)):
                    if (((x-x0)**2 + (y-y0)**2) <= args.radius**2) and (x<x_max and y<y_max) and (x>0 and y >0):
                        arr[x,y,cts_idx] = 1

        np.save(SAVE_PATH.joinpath(s_roi+'.npy'), arr)

    

