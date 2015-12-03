"""
Script to do SVD on the covariance matrix of the MASKED voxel by time matrix.
Spits out the first 5 components for each subject.

Run with: 
    python get_pcs_script.py

"""

import numpy as np
import numpy.linalg as npl
import nibabel as nib
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Relative paths to project and data. 
project_path          = "../../../"
path_to_data          = project_path+"data/ds009/"
location_of_images    = project_path+"images/"
location_of_functions = project_path+"code/utils/functions/" 
behav_suffix           = "/behav/task001_run001/behavdata.txt"

sys.path.append(location_of_functions)
from Image_Visualizing import make_mask

# List of subject directories.
#sub_list = os.listdir(path_to_data)
sub_list = ['sub002', 'sub003', 'sub014'] # Just 3 for now.
# Initialize list to store principal components for each subject.
pcs = []

# Loop through all the subjects. 
for name in sub_list:
    # amount of beginning TRs not standardized at 6
    behav=pd.read_table(path_to_data+name+behav_suffix,sep=" ")
    num_TR = float(behav["NumTRs"])
    
    # Load image data.
    img = nib.load(path_to_data+ name+ "/BOLD/task001_run001/bold.nii.gz")
    data = img.get_data()
    data = data.astype(float) 

    # Load mask.
    mask = nib.load(path_to_data+ name+'/anatomy/inplane001_brain_mask.nii.gz')
    mask_data = mask.get_data()

    # Drop the appropriate number of volumes from the beginning. 
    first_n_vols=data.shape[-1]
    num_TR_cut=int(first_n_vols-num_TR)
    data = data[...,num_TR_cut:] 

    # Now fit a mask to the 3-d image for each time point.
    my_mask = np.zeros(data.shape)
    for i in range(my_mask.shape[-1]): 
         my_mask[...,i] = make_mask(data[...,i], mask_data, fit=True)
    
    # Reshape stuff to 2-d (voxel by time) and mask the data. 
    # This should cut down the number of volumes by more than 50%.
    my_mask_2d = my_mask.reshape((-1,my_mask.shape[-1]))
    data_2d = data.reshape((-1,data.shape[-1]))
    masked_data_2d = data_2d[my_mask_2d.sum(1) != 0,:]

    # Subtract means over voxels (columns).
    masked_data_2d = masked_data_2d - np.mean(masked_data_2d, 0)

    # Subtract means over time (rows)
    masked_data_2d = masked_data_2d - np.mean(masked_data_2d, axis=1)[:, None]

    # PCA analysis on MASKED data: 
    # Do SVD on the time by time matrix and get explained variance.
    U_masked, S_masked, VT_masked = npl.svd(masked_data_2d.T.dot(masked_data_2d))
    pcs.append(masked_data_2d.dot(U_masked[:,:5]))
    
