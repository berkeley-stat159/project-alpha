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

    # Load conditions. 
    condition_location = path_to_data+ name+ "/model/model001/onsets/task001_run001/"
    cond1=np.loadtxt(condition_location+"cond001.txt")
    cond2=np.loadtxt(condition_location+"cond002.txt")
    cond3=np.loadtxt(condition_location+"cond003.txt")
    cond_all=np.row_stack((cond1,cond2,cond3))
    cond_all=sorted(cond_all,key= lambda x:x[0])
    np.savetxt(condition_location+"cond_all.txt",cond_all)
    cond_all = np.loadtxt(condition_location+"cond_all.txt")

    # Drop the appropriate number of volumes from the beginning. 
    first_n_vols=data.shape[-1]
    num_TR_cut=int(first_n_vols-num_TR)
    num_cut[i]=num_TR_cut
    i+=1
    data = data[...,num_TR_cut:] 

    # Set up a design matrix that takes into account noise corrections.
    n_vols = data.shape[-1]
    TR = 2
    all_tr_times = np.arange(n_vols) * TR

    y_mean=mean_underlying_noise(data)
    X_mean, MRSS_mean, fitted_mean,residuals_mean=fourier_predict_underlying_noise(y_mean,3)

    X = np.ones((n_vols,4))
    X[:,1]=convolution_specialized(cond_all[:,0],np.ones(len(cond_all)),hrf_single,all_tr_times)
    drift= np.linspace(-1,1,num=X.shape[0])
    X[:,2]=drift
    X[:,3]=fitted_mean

    # Run PCA on the covariance matrix and plot explained variance.
    pca = PCA(n_components=5)
    pca.fit(np.cov(X))
    exp_var = pca.explained_variance_ratio_ 
    plt.plot(exp_var)
    plt.savefig(location_of_images+'pca'+name+'.png')
    plt.close()
