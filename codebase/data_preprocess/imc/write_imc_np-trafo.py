import numpy as np
import pandas as pd
import json
import math
import os
import argparse
import cv2
import tifffile

from codebase.utils.constants import *
from codebase.utils.raw_utils import *
from codebase.utils.dataset_utils import get_patient_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess and write IMC as numpy arrays.')
    parser.add_argument('--save_path', type=str, required=False, default='/cluster/work/grlab/projects/projects2021-multivstain/data/tupro/', help='Path to save the processed data')
    parser.add_argument('--input_path', type=str, required=False, default='/cluster/work/grlab/projects/projects2021-multivstain/data/tupro/binary_imc_rois_raw/', help='Path to load raw data from')
    parser.add_argument('--sample_list', type=str, required=False, default='', help='List of samples to parse, separated by comma.')
    parser.add_argument('--overwrite', type=str2bool, required=False, default=False, help='Whether to overwrite if output files exist')
    parser.add_argument('--arcsinh', type=str2bool, required=False, default=False)
    parser.add_argument('--clip99', type=str2bool, required=False, default=False, help='Whether to clip data to 99th percentile')
    parser.add_argument('--median_blur_ksize', type=int, required=False, default=0, help='Kernel size of median blur, if 0, then no median blurring is applied')
    parser.add_argument('--use_cell_mask', type=str2bool, required=False, default=False, help='Whether to set signal outside of cells to 0')
    parser.add_argument('--smooth_per_cell', type=str2bool, required=False, default=False, help='Whether to average protein values across pixels within a cell')
    parser.add_argument('--protein_list_name', type=str, required=False, default="PROTEIN_LIST_MVS", help='Name of the protein list to use {PROTEIN_LIST_MVS, PROTEIN_LIST, PROTEIN_LIST_FULL, PROTEIN_LIST_FULL_IR}, consult code.utils.constants.py')
    
    args = parser.parse_args()
    
    # protein list
    protein_list = get_protein_list_by_name(args.protein_list_name)

    # sample list
    if args.sample_list == '':
        sel_samples = MVS_SAMPLES
    else:
        sel_samples = args.sample_list.split(',')

    print('#samples: '+str(len(sel_samples)))
    
    sample_patient_df = get_patient_samples(data_dir = '/cluster/work/tumorp/data_repository/',
                                            phase='study', indication='melanoma')
    sample_patient_dict = dict(zip(sample_patient_df['sample'], sample_patient_df['patient']))
    
    if args.arcsinh:
        def trafo(x): return np.log(x + np.sqrt(x**2 + 1))
    else:
        def trafo(x): return x
        
    if args.smooth_per_cell or args.use_cell_mask:
        mask_file_map = pd.read_csv('/cluster/work/grlab/projects/projects2021-multivstain/meta/cell_mask_paths-mvs_samples.csv')
        def smooth_pixels(x):
            return np.array([mean_dict[y] for y in x])
        
    files = os.listdir(args.input_path)
    failed_roi = []
    for i,s in enumerate(sel_samples):
        rois = sorted([x.split('_')[-1].split('.')[0] for x in files if s in x])
        
        imc_sample_np_dict = dict()
        imc_sample_mask_dict = dict()
        exists_roi = []
        for roi_id in rois:
            print(s, roi_id)
            save_file = os.path.join(args.save_path, s + "_" + roi_id + ".npy")
                    
            if not args.overwrite and os.path.exists(save_file):
                exists_roi.append(roi_id)
                print('Data for sample '+s+' and roi '+roi_id+' already exist -> skipping')
                continue
                
            imc_roi_np = np.load(os.path.join(args.input_path, s+'_'+roi_id+'.npy'))
            print('Loaded raw data')
                
            if args.smooth_per_cell or args.use_cell_mask:
                #print('Cell masks are not yet available, not smoothing')
                tiff_path = mask_file_map.loc[mask_file_map.sample_roi==s+'_'+roi_id,'mask_file_path'].values[0]
                #tiff_path = IMC_MASK_PATH+mask_fname
                mask = tifffile.imread((tiff_path))
                # need to flip the mask as the IMC ROIs were flipped in preprocessing steps
                mask = np.flip(mask,0) #equivalent to np.flipud(mask)
                # need to cut the mask to square 1000x1000 as in IMC (protein) data preprocessing
                if imc_roi_np.shape[0]!=mask.shape[0]:
                    print(s+'_'+roi_id+' could not proceed')
                    failed_roi.append(s+'_'+roi_id)
                    continue
                mask = mask[0:1000,0:1000]
                if args.smooth_per_cell:
                    unq,ids,count = np.unique(mask.flatten(),return_inverse=True,return_counts=True)
                    # average pixel signal within cells (per channel)
                    print('Loaded cell mask')
                    for i in range(imc_roi_np.shape[2]):
                        try:
                            out = np.column_stack((unq,np.bincount(ids,imc_roi_np[:,:,i].flatten())/count))
                        except:
                            import pdb; pdb.set_trace()
                        mean_dict = dict(zip(out[:,0].astype(int), out[:,1]))
                        imc_roi_np[:,:,i] = np.apply_along_axis(smooth_pixels, 0, mask)
                    print('Finished smoothing data')

            if args.clip99:
                # save ROI data in per-sample array dictionary
                imc_sample_np_dict[roi_id] = imc_roi_np
                if args.use_cell_mask:
                    imc_sample_mask_dict[roi_id] = mask
            else:
                # if no sample-wise clipping, then can proceed directly with trafo (and filtering) and save
                if args.median_blur_ksize>0:
                    imc_roi_np = cv2.medianBlur(np.float32(imc_roi_np), args.median_blur_ksize)
                imc_roi_np = trafo(imc_roi_np)
                
                if args.use_cell_mask:
                    for i in range(imc_roi_np.shape[2]):
                        imc_roi_np[:,:,i] = imc_roi_np[:,:,i]*(mask>0)
                
                if not args.overwrite and os.path.exists(save_file):
                    continue
                print(save_file)
                np.save(save_file, imc_roi_np)

        
        if args.clip99:
            print('clipping option')
            if len(imc_sample_np_dict)<1:
                continue
            elif len(imc_sample_np_dict)<len(exists_roi):
                print('Not all ROIs for sample '+s+' were processed in this run - cannot compute q99; please rerun with overwrite=True')
                continue
            else:
                # extract 99th percentile per channel
                imc_sample_np = np.concatenate([imc_sample_np_dict[x].transpose(2,0,1).reshape(len(protein_list),-1) for x in imc_sample_np_dict.keys()], axis=1)
                q99 = np.percentile(imc_sample_np, 99, axis=1)


            # (clip and arsinh transform) and save data per ROI
            for roi_id in rois:
                if len(imc_sample_np_dict.keys())<len(rois):
                    print('Could not proceed for sample '+s)
                    continue
                imc_roi_np = np.clip(imc_sample_np_dict[roi_id], a_min=0, a_max=q99)
                imc_roi_np = trafo(imc_roi_np)
                
                if args.use_cell_mask:
                    if roi_id in imc_sample_mask_dict.keys():
                        for i in range(imc_roi_np.shape[2]):
                            imc_roi_np[:,:,i] = imc_roi_np[:,:,i]*(imc_sample_mask_dict[roi_id]>0)
                    else:
                        print('Could not find aligned mask for ROI '+s+'_'+roi_id)
                        continue
                
                save_file = os.path.join(args.save_path, s + "_" + roi_id + ".npy")
                if not args.overwrite and os.path.exists(save_file):
                    continue
                print(save_file)
                np.save(save_file, imc_roi_np)

    if len(failed_roi)>0:            
        print(failed_roi)
