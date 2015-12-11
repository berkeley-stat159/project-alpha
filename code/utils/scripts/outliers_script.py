"""
Script to identify outliers for each subject. 
Compares the mean MRSS values from running GLM on the basic np.convolve convolved time course, 
before and after dropping the outliers. 

"""

import numpy as np
import nibabel as nib
import os
import sys
import pandas as pd

# Relative paths to project and data. 
project_path          = "../../../"
path_to_data          = project_path+"data/ds009/"
location_of_functions = project_path+"code/utils/functions/" 
behav_suffix           = "/behav/task001_run001/behavdata.txt"

sys.path.append(location_of_functions)

from stimuli import events2neural
from event_related_fMRI_functions import hrf_single
from outliers import *

sub_list = os.listdir(path_to_data)
sub_list = [i for i in sub_list if 'sub' in i]


# List to store the mean MRSS values before and after outlier removal
MRSSvals = []

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

    # Drop the appropriate number of volumes from the beginning. 
    first_n_vols=data.shape[-1]
    num_TR_cut=int(first_n_vols-num_TR)
    num_cut[i]=num_TR_cut
    i+=1
    data = data[...,num_TR_cut:] 

    # initial needed values
    TR = 2
    tr_times = np.arange(0, 30, TR)
    hrf_at_trs = np.array([hrf_single(x) for x in tr_times])
    n_vols=data.shape[-1]

    # creating the .txt file for the events2neural function
    cond_all=np.row_stack((cond1,cond2,cond3))
    cond_all=sorted(cond_all,key= lambda x:x[0])
    np.savetxt(condition_location+"cond_all.txt",cond_all)

    # Get the convolved time course from np.convolve
    neural_prediction = events2neural(condition_location+"cond_all.txt",TR,n_vols)
    convolved = np.convolve(neural_prediction, hrf_at_trs) # hrf_at_trs sample data
    np_hrf=convolved[:n_vols]

    # mean MRSS values before and after dropping outliers. 
    MRSSvals.append((name,) + compare_outliers(data, np_hrf))

#np.savetxt("outlierMRSSvals.txt",MRSSvals)
print(MRSSvals)

'''
By and large, mean MRSS doesn't seem to shift much before and after dropping outliers. 
''' 
