import numpy as np
import pandas as pd
from skimage import draw
from pathlib import Path
import argparse
import os
import json

import sys 
root_code = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.insert(0, root_code)

from codebase.utils.constants import *
from codebase.utils.raw_utils import str2bool
from codebase.utils.eval_utils import get_protein_list, get_last_epoch_w_imgs, get_best_epoch_w_imgs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate signal across pseudocells (circles around nuclei centroids).')
    parser.add_argument('--project_path', type=str, required=False, default='/raid/sonali/project_mvs/', help='Path where all data, results etc for project reside')
    parser.add_argument('--submission_id', type=str, required=True, default=None, help='Job submission_id')
    parser.add_argument('--epoch', type=str, required=False, default='last', help='Which epoch to use (e.g.,2 or 2-3), if last then takes the last one found, if best then searching for chkpt_selection info')
    parser.add_argument('--level', type=int, required=False, default=2, help='Which resolution to use {2,4,6}')
    parser.add_argument('--data_set', type=str, required=False, default="test", help='Which set from split to use {test, valid, train}')
    parser.add_argument('--sel_roi_name', type=str, required=False, default=None, help='Selected ROI to perform eval on; if not specified, then apply to all ROIs for which predicted images exist')
    parser.add_argument('--radius', type=int, required=False, default=5, help='Radius of the circle to be drawn around the XY cell coordinates')
    parser.add_argument('--overwrite', type=str2bool, required=False, default=False, help='Whether to overwrite if output files exist')
    args = parser.parse_args()

    # python agg_pred_data_masks.py --submission_id='utdfvuvw_dgm4h_pyramidp2p_selected_snr_pseudo-multiplex_dgm4h' --epoch='250K' 

    # Selection of epoch based on best/last or specified
    if args.epoch == 'last':
        epoch = get_last_epoch_w_imgs(args.project_path, args.submission_id)
    elif args.epoch =='best':
        epoch = get_best_epoch_w_imgs(args.project_path, args.submission_id)
    else:
        epoch = 'step_'+args.epoch
    
    # set the paths
    PROJECT_PATH = Path(args.project_path)
    RESULTS_DIR = PROJECT_PATH.joinpath('results')
    INPUT_PATH = RESULTS_DIR.joinpath(args.submission_id,args.data_set+'_images', epoch, 'level_'+str(args.level))
    SAVE_PATH = RESULTS_DIR.joinpath(args.submission_id,args.data_set+'_scdata', epoch, 'level_'+str(args.level))
    if not os.path.exists(SAVE_PATH):
        SAVE_PATH.mkdir(parents=True, exist_ok=False)
        existing_rois = []
    else:
        existing_rois = [] if args.overwrite else [x.split('.')[0] for x in os.listdir(SAVE_PATH)]
    
    # get job arguments
    job_args = json.load(open(RESULTS_DIR.joinpath(args.submission_id, 'args.txt')))
    # get sample_roi list
    cv = json.load(open(PROJECT_PATH.joinpath(CV_SPLIT_ROIS_PATH)))
    sample_rois = cv[job_args['cv_split']][args.data_set]
    protein_list = get_protein_list(job_args['protein_set'])
    
    # load XY cell coordinates
    he_coords = pd.read_csv(PROJECT_PATH.joinpath('meta/tupro/hovernet/hovernet_nuclei-coordinates_all-samples.csv'))
    # adjust to desired resolution
    he_coords['X'] = he_coords['X']//(2**(args.level))
    he_coords['Y'] = he_coords['Y']//(2**(args.level))
    
    # adjust radius to desired resolution
    radius = args.radius*(4//(2**(args.level)))
        
    for m, s_roi in enumerate(sample_rois):
        
        if s_roi in existing_rois:
            #print(s_roi+' already in the data, skipping')
            continue
        else:
            print(m, s_roi)
            imc_roi_np = np.load(INPUT_PATH.joinpath(s_roi+'.npy'), mmap_mode='r')
            x_max, y_max, n_channels = imc_roi_np.shape
            df = he_coords.loc[he_coords.sample_roi==s_roi,:]

            sc_df = pd.DataFrame(index=df.index.to_list(), columns=protein_list)
            mask = np.zeros((x_max, y_max,1))
            for i in range(df.shape[0]):
                object_df = df.iloc[i,:]
                x0 = object_df['X']
                y0 = object_df['Y']
                rr, cc = draw.circle_perimeter(x0, y0, radius=radius, shape=mask.shape, method='andres')
                # circle contour
                mask[rr, cc,0] = i
                # fill the circle
                for x in range(abs(x0-radius),(x0+radius)):
                    for y in range(abs(y0-radius),(y0+radius)):
                        if (((x-x0)**2 + (y-y0)**2) <= radius**2) and (x<x_max and y<y_max) and (x>0 and y >0):
                            mask[x,y,0] = i

            unq,ids,count = np.unique(mask.flatten(),return_inverse=True,return_counts=True)
            # average pixel signal within cells (per channel)
            for j,prot in enumerate(protein_list):
                try:
                    out = np.column_stack((unq,np.bincount(ids,imc_roi_np[:,:,j].flatten())/count))
                except:
                    import pdb; pdb.set_trace()
                mean_dict = dict(zip(out[:,0].astype(int), out[:,1]))
                for i in mean_dict.keys():
                    sc_df.loc[df.index.to_list()[i],prot] = mean_dict[i]
            sc_df['sample_roi'] = s_roi
            sc_df = sc_df.merge(df.loc[:,['X','Y']], left_index=True, right_index=True, how='left')
            sc_df['radius'] = radius

            sc_df.to_csv(SAVE_PATH.joinpath(s_roi+'.tsv'), sep='\t')
    
    
