# code for running in terminal 
import os 
import numpy as np 
import cv2 
import staintools 
import glob
import matplotlib.pyplot as plt

base_path = '/raid/sonali/project_mvs/data/tupro/binary_he_rois'
he_paths = glob.glob(base_path + '/*.npy')
len(he_paths), he_paths[0:3]

save_path =  '/raid/sonali/project_mvs/data/tupro/binary_he_rois_normalised' 
if not os.path.exists(save_path): 
    os.mkdir(save_path)

for he_path in he_paths: 
    he_img = (np.load(he_path)*255).astype(np.uint8)
    he_img = (staintools.LuminosityStandardizer.standardize(he_img))
    he_img = (he_img/255).astype(np.float32)                            
    np.save(os.path.join(save_path, he_path.split('/')[-1].split('.npy')[0] + '.npy'), he_img)
    print(he_path)



