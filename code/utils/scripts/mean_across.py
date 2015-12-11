""" Averages t-values per voxel across all subjects using simple linear regression
This function is used later for our final scripts

Run with: 
    python mean_across.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "functions"))

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from stimuli import events2neural
from event_related_fMRI_functions import hrf_single, convolution_specialized
from hypothesis import t_stat
from Image_Visualizing import present_3d, make_mask

pathtodata = "../../../data/ds009/"
sub_list = os.listdir(pathtodata)
sub_list = [i for i in sub_list if 'sub' in i]

#Create empty array for the t-values per voxel per subject
t_mean = np.zeros((64, 64, 34,24))

#loop through all of the subjects
for i in sub_list:
    img = nib.load(pathtodata+ i+ "/BOLD/task001_run001/bold.nii.gz")
    data = img.get_data()
    data = data[...,6:] 
    
    # Suppose that TR=2. We know this is not a good assumption.
    # Also need to look into the hrf function. 
    cond1=np.loadtxt(pathtodata+ i+ "/model/model001/onsets/task001_run001/cond001.txt")
    cond2=np.loadtxt(pathtodata+ i+ "/model/model001/onsets/task001_run001/cond002.txt")
    cond3=np.loadtxt(pathtodata+ i+ "/model/model001/onsets/task001_run001/cond003.txt")
    
    TR = 2
    tr_times = np.arange(0, 30, TR)
    hrf_at_trs = np.array([hrf_single(x) for x in tr_times])
    n_vols=data.shape[-1]

    # creating the .txt file for the events2neural function
    cond_all=np.row_stack((cond1,cond2,cond3))
    cond_all=sorted(cond_all,key= lambda x:x[0])
    np.savetxt(pathtodata+ i+ "/model/model001/onsets/task001_run001/cond_all.txt",cond_all)

    neural_prediction = events2neural(pathtodata+ i+ "/model/model001/onsets/task001_run001/cond_all.txt",TR,n_vols)
    convolved = np.convolve(neural_prediction, hrf_at_trs) # hrf_at_trs sample 
    
    N = len(neural_prediction)  # N == n_vols == 173
    M = len(hrf_at_trs)  # M == 12
    np_hrf=convolved[:N]
    
    B,t,df,p = t_stat(data, np_hrf, np.array([0,1]))

     #Simple mask function
    mask = nib.load(pathtodata+i+'/anatomy/inplane001_brain_mask.nii.gz')
    mask_data = mask.get_data()
    
    t_mean[...,int(i[-1])] = make_mask(np.reshape(t,(64,64,34)), mask_data, fit=True)


    
final = present_3d(np.mean(t_mean,axis=3))


#######################
# Plot the results    #
#######################
plt.imshow(final,interpolation='nearest', cmap='seismic')
plt.title("Mean T-Statistic Value Across 25 Subjects")

zero_out=max(abs(np.min(final)),np.max(final))
plt.clim(-zero_out,zero_out)
plt.colorbar()
plt.savefig("../../../paper/images/hypothesis_testing.png")
plt.close()





