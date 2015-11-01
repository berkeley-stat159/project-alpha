import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "functions"))

import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
import nibabel as nib
from glm import glm, glm_diagnostics
from stimuli import events2neural
from event_related_fMRI_functions import convultion

pathtodata = "../../../data/ds009/"
sub_list = os.listdir(pathtodata)[1:]

rss_mean = np.zeros((64, 64, 34,24))

def hrf(times):
    """ Return values for HRF at given times """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6
    
for i in os.listdir(pathtodata)[1:]:
    img = nib.load(pathtodata+ i+ "/BOLD/task001_run001/bold.nii.gz")
    data = img.get_data()
    data = data[...,6:] 
    
    # Suppose that TR=2. We know this is not a good assumption.
    # Also need to look into the hrf function. 
    TR = 2
    tr_times = np.arange(0, 30, 2)
    hrf_at_trs = hrf(tr_times)
    n_vols = data.shape[-1]
    events = events2neural(pathtodata+ i+ "/model/model001/onsets/task001_run001/cond001.txt", TR, n_vols)
    events = np.abs(events-1) # Swapping 0s and 1s. 
    convolved = np.convolve(events, hrf_at_trs)
    n_to_remove = len(hrf_at_trs) - 1
    convolved = convolved[:-n_to_remove]

    # Now get the estimated coefficients and design matrix for doing
    # regression on the convolved time course. 
    B, X = glm(data, convolved)

    # Some diagnostics. 
    MRSS, fitted, residuals = glm_diagnostics(B, X, data)
    
    rss_mean[...,int(i[-1])] = MRSS
    
print(np.mean(rss_mean,axis=3))
    



