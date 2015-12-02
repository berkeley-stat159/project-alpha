"""
Script to do SVD on the covariance matrix of the voxel by time matrix.

Run with: 
    python pca_script.py

"""

import numpy as np
import nibabel as nib
import os
import sys
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Relative paths to project and data. 
project_path          = "../../../"
path_to_data          = project_path+"data/ds009/"
location_of_images    = project_path+"images/"
location_of_functions = project_path+"code/utils/functions/" 
behav_suffix           = "/behav/task001_run001/behavdata.txt"

sys.path.append(location_of_functions)

from Image_Visualizing import make_mask

sub_list = os.listdir(path_to_data)

# saving to compare number of cuts in the beginning
num_cut=np.zeros(len(sub_list))
i=0

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
    num_cut[i]=num_TR_cut
    i+=1
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

    # Subtract means from columns.
    data_2d = data_2d - np.mean(data_2d, 0)
    masked_data_2d = masked_data_2d - np.mean(masked_data_2d, 0)

    # PCA analysis on unmasked data: 
    # Do SVD for the first 20 values of the time by time matrix and plot explained variance.
    pca = PCA(n_components=20)
    pca.fit(data_2d.T.dot(data_2d))
    exp_var = pca.explained_variance_ratio_ 
    plt.plot(range(1,21), exp_var)
    plt.savefig(location_of_images+'pca'+name+'.png')
    plt.close()

    # PCA analysis on MASKED data: 
    # Do SVD for the first 20 values of the time by time matrix and plot explained variance.
    pca_masked = PCA(n_components=20)
    pca_masked.fit(masked_data_2d.T.dot(masked_data_2d))
    exp_var_masked = pca_masked.explained_variance_ratio_ 
    plt.plot(range(1,21), exp_var_masked)
    plt.savefig(location_of_images+'maskedpca'+name+'.png')
    plt.close()
