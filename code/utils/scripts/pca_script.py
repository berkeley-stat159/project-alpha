"""
Script to identify outliers for each subject. Compares the mean MRSS values from running GLM on the basic np.convolve convolved time course, before and after dropping the outliers. 

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

from event_related_fMRI_functions import hrf_single, convolution_specialized
from noise_correction import mean_underlying_noise, fourier_predict_underlying_noise

sub_list = os.listdir(path_to_data)[0:2]

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

    # Drop the appropriate number of volumes from the beginning. 
    first_n_vols=data.shape[-1]
    num_TR_cut=int(first_n_vols-num_TR)
    num_cut[i]=num_TR_cut
    i+=1
    data = data[...,num_TR_cut:] 
    data_2d = data.reshape((-1,data.shape[-1]))

    # Run PCA on the covariance matrix and plot explained variance.
    pca = PCA(n_components=20)
    pca.fit(data_2d.T.dot(data_2d))
    exp_var = pca.explained_variance_ratio_ 
    plt.plot(range(1,21), exp_var)
    plt.savefig(location_of_images+'pca'+name+'.png')
    plt.close()
