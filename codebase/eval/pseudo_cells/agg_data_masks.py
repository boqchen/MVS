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
    parser = argparse.ArgumentParser(description='Aggregate signal across pseudocells (circles around nuclei centroids).')
    parser.add_argument('--project_path', type=str, required=False, default='/raid/sonali/project_mvs/', help='path where all data, results etc for project reside')
    parser.add_argument('--sample_list', type=str, required=False, default='', help='List of samples to parse, separated by comma')
    parser.add_argument('--protein_list', type=str, required=False, default='PROTEIN_LIST_MVS', help='List of proteins')
    parser.add_argument('--radius', type=int, required=False, default=5, help='Radius of the circle to be drawn around the XY cell coordinates')
    parser.add_argument('--imc_prep_seq', type=str, required=False, default='raw_clip99_arc_otsu3', help='Sequence of data preprocessing steps (needed to select form which folder to load the data)')
    parser.add_argument('--standardize_imc', type=str2bool, required=False, default=False, help='Whether to standardize IMC data using cohort mean and stddev per channel')
    parser.add_argument('--scale01_imc', type=str2bool, required=False, default=False, help='Whether to scale IMC data to [0,1] interval per channel using cohort stats')
    parser.add_argument('--cv_split', type=str, required=False, default='split3', help='Selected CV split, if None then the splitting used for the report is used')
    parser.add_argument('--overwrite', type=str2bool, required=False, default=False, help='Whether to overwrite if output files exist')
    args = parser.parse_args()

    PROJECT_PATH = Path(args.project_path)
    INPUT_PATH = Path(get_imc_roi_storage(args.project_path+DATA_DIR, args.imc_prep_seq, 
                                      standardize_imc=args.standardize_imc, scale01_imc=args.scale01_imc,
                                      cv_split=args.cv_split))
    imc_prep_name = args.imc_prep_seq
    if (args.standardize_imc or args.scale01_imc):
        if args.standardize_imc:
            imc_prep_name = imc_prep_name+'_std'
        if args.scale01_imc:
            imc_prep_name = imc_prep_name+'_minmax'
        imc_prep_name = imc_prep_name+'_'+args.cv_split
    save_fname = '-'.join(['agg_masked_data', imc_prep_name, 'r'+str(args.radius)])
    
    cts = pd.read_csv(PROJECT_PATH.joinpath('data/tupro/imc_updated/coldata.tsv'), sep='\t', index_col=[0])
    
    if args.sample_list != '':
        sample_rois = args.sample_list.split(',')
    else:
        # load from cv splits
        cv = json.load(open(PROJECT_PATH.joinpath(CV_SPLIT_ROIS_PATH)))
        sample_rois = cv[args.cv_split]['train']
        sample_rois.extend(cv[args.cv_split]['valid'])
        sample_rois.extend(cv[args.cv_split]['test'])

    protein_list = get_protein_list_by_name(protein_list_name=args.protein_list)

    if os.path.exists(PROJECT_PATH.joinpath(DATA_DIR, 'imc_updated',save_fname+'.tsv')) and not args.overwrite:
        sc_df_all = pd.read_csv(PROJECT_PATH.joinpath(DATA_DIR, 'imc_updated',save_fname+'.tsv'), sep='\t', index_col=[0])
        existing_rois = sorted(sc_df_all.sample_roi.unique())
        print('Data for '+str(len(existing_rois))+' ROIs found')
    else:
        sc_df_all = pd.DataFrame()
        existing_rois = []
        
    for m, s_roi in enumerate(sample_rois):
        
        if s_roi in existing_rois:
            #print(s_roi+' already in the data, skipping')
            continue
        else:
            print(m, s_roi)
            imc_roi_np = np.load(INPUT_PATH.joinpath(s_roi+'.npy'), mmap_mode='r')
            imc_roi_np = imc_roi_np[:,:,[protein2index[x] for x in protein_list]]
            x_max, y_max, n_channels = imc_roi_np.shape
            df = cts.loc[cts.sample_roi==s_roi,:]

            sc_df = pd.DataFrame(index=df.index.to_list(), columns=protein_list)
            mask = np.zeros((x_max, y_max,1))
            for i in range(df.shape[0]):
                object_df = df.iloc[i,:]
                # the coordinates of the ROIs are flipped wrt cts
                x0 = object_df['Y']
                y0 = object_df['X']
                rr, cc = draw.circle_perimeter(x0, y0, radius=args.radius, shape=mask.shape, method='andres')
                # circle contour
                mask[rr, cc,0] = i
                # fill the circle
                for x in range(abs(x0-args.radius),(x0+args.radius)):
                    for y in range(abs(y0-args.radius),(y0+args.radius)):
                        if (((x-x0)**2 + (y-y0)**2) <= args.radius**2) and (x<x_max and y<y_max) and (x>0 and y >0):
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
            coords = df.loc[:,['X','Y']]
            # the coordinates of the ROIs are flipped wrt cts
            coords.columns = ['Y','X']
            sc_df = sc_df.merge(coords, left_index=True, right_index=True, how='left')
            sc_df['radius'] = args.radius
            sc_df_all = pd.concat([sc_df_all,sc_df])

            sc_df_all.to_csv(PROJECT_PATH.joinpath(DATA_DIR, 'imc_updated', save_fname+'.tsv'), sep='\t')


