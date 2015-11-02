import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "functions"))

import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
import nibabel as nib
from glm import glm, glm_diagnostics
from stimuli import events2neural
from event_related_fMRI_functions import hrf_single, convolution_specialized

pathtodata = "../../../data/ds009/"
sub_list = os.listdir(pathtodata)[1:]

rss_mean = np.zeros((64, 64, 34,24))


    
for i in os.listdir(pathtodata)[1:]:
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
    convolved = np.convolve(neural_prediction, hrf_at_trs) # hrf_at_trs sample data
    N = len(neural_prediction)  # N == n_vols == 173
    M = len(hrf_at_trs)  # M == 12
    np_hrf=convolved[:N]
   
    # Now get the estimated coefficients and design matrix for doing
    # regression on the convolved time course. 
    B, X = glm(data, np_hrf)

    MRSS, fitted, residuals = glm_diagnostics(B, X, data)
    
    rss_mean[...,int(i[-1])] = MRSS
    
print(np.mean(rss_mean,axis=3))
    



