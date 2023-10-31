import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import os
import json
from scipy.stats.stats import pearsonr, spearmanr

from codebase.utils.constants import *
from codebase.utils.raw_utils import str2bool

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare nuclei density.')
    parser.add_argument('--project_path', type=str, required=False, default='/raid/sonali/project_mvs/', help='Path where all data, results etc for project reside')
    parser.add_argument('--metric', type=str, required=False, default='pcorr', help='Which metric to use for comparison {pcorr, spcorr}')
    parser.add_argument('--resolutions', type=str, required=False, default='1,4,16,32,64,75,128,256', help='Which resolution in pixels to use')
    parser.add_argument('--axmax', type=int, required=False, default=1000, help='Maximal size of ROI to use {1000,1024} to ease pixel interpretation')
    parser.add_argument('--cv_split', type=str, required=False, default='split3', help='Selected CV split, if None then the splitting used for the report is used')
    parser.add_argument('--data_set', type=str, required=False, default="all", help='Which set from split to use {test, valid, train, all}; if all then all three splits are used')
    parser.add_argument('--density', type=str2bool, required=False, default=True, help='Whether to use nuclei density or counts')
    parser.add_argument('--overwrite', type=str2bool, required=False, default=False, help='Whether to overwrite if output files exist')
    args = parser.parse_args()
    
    PROJECT_PATH = Path(args.project_path)
    cv = json.load(open(PROJECT_PATH.joinpath(CV_SPLIT_ROIS_PATH)))
    if args.data_set == 'all':
        sample_rois = cv[args.cv_split]['train']
        sample_rois.extend(cv[args.cv_split]['valid'])
        sample_rois.extend(cv[args.cv_split]['test'])
    else:
        sample_rois = cv[args.cv_split][args.data_set]
  
    resolutions = [int(x) for x in args.resolutions.split(',')]
    
    # Load nuclei coordinates
    he_coords = pd.read_csv(PROJECT_PATH.joinpath('meta/hovernet/hovernet_nuclei-coordinates_all-samples.csv'))
    he_coords = he_coords.loc[he_coords.sample_roi.isin(sample_rois),:]
    # Adjust to IMC resolution
    he_coords['X'] = he_coords['X']//4
    he_coords['Y'] = he_coords['Y']//4
    imc_coords = pd.read_csv(PROJECT_PATH.joinpath('data/tupro/imc_updated/coldata.tsv'), sep='\t')
    imc_coords = imc_coords.loc[imc_coords.sample_roi.isin(sample_rois),:]

    pcorr_df_all = dict()
    for desired_resolution_px in resolutions:
        print(desired_resolution_px)

        n_bins = args.axmax//desired_resolution_px
        x_bins = np.linspace(0, 1000, n_bins+1)
        y_bins = np.linspace(0, 1000, n_bins+1)

        pcorr_df = pd.DataFrame(index=sample_rois, columns=[args.metric])
        for s_roi in sample_rois:
            # HE nuclei density
            df_roi = he_coords.loc[he_coords.sample_roi==s_roi,:]
            density_he, _, _ = np.histogram2d(df_roi['X'],df_roi['Y'], [x_bins, y_bins], density=args.density)
            # IMC nuclei density
            df_roi = imc_coords.loc[imc_coords['sample_roi']==s_roi,:]
            density_imc, _, _ = np.histogram2d(df_roi['X'],df_roi['Y'], [x_bins, y_bins], density=args.density)
            # plot density
            if args.metric == 'pcorr':
                pcorr_df.loc[s_roi, args.metric] = pearsonr(density_he.flatten(), density_imc.flatten())[0]
            elif args.metric == 'spcorr':
                pcorr_df.loc[s_roi, args.metric] = spearmanr(density_he.flatten(), density_imc.flatten())[0]
                
        pcorr_df_all[desired_resolution_px] = pcorr_df
        #pcorr_df_all[str(desired_resolution_px)+'_'+str(n_bins**2)] = pcorr_df

    pcorr_df_all = pd.concat(pcorr_df_all)
    pcorr_df_all = pcorr_df_all.reset_index()
    pcorr_df_all.columns = ['resolution_bins', 'sample_roi', args.metric]
    pcorr_df_all = pcorr_df_all.pivot(index='sample_roi', columns='resolution_bins', values=args.metric)
    pcorr_df_all.columns.name = None
    pcorr_df_all.columns = [args.metric+'_'+str(x) for x in pcorr_df_all.columns]
    pcorr_df_all.head(2)

    cv_split_name = args.cv_split+'_'+args.data_set if args.data_set!='all' else 'all'
    save_fname = 'nuclei_density' if args.density else 'nuclei_counts'
    save_fname = '-'.join([save_fname,'he_imc', cv_split_name, args.metric, 'max'+str(args.axmax)+'.tsv'])
    if args.overwrite == True:
        pcorr_df_all.to_csv(PROJECT_PATH.joinpath('meta', 'nuclei_density', save_fname), sep='\t')    
    else:
        if os.path.exists(PROJECT_PATH.joinpath('meta', 'nuclei_density', save_fname))==False:
            pcorr_df_all.to_csv(PROJECT_PATH.joinpath('meta', 'nuclei_density', save_fname), sep='\t')