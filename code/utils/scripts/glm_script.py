""" Script for GLM functions.
Run with: 
    python glm_script.py
"""

# Loading modules.
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
import nibabel as nib
import os
import sys

# Relative path to subject 1 data
pathtodata = "../../../data/ds009/sub001/"

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "functions"))

# Load events2neural from the stimuli module.
from stimuli import events2neural

# Load our GLM functions. 
from glm import glm, glm_diagnostics

# Load the image data for subject 1.
img = nib.load(pathtodata+"BOLD/task001_run001/bold.nii.gz")
data = img.get_data()
data = data[...,6:] # Knock off the first 6 observations.


""" This stuff is a bit of a hack to get a naive convolved time 
course using TR = 2. 
Hopefully, we'll have a better, more integrated way of getting 
the convolutions soon, with its own script for generating them. 
"""
# hrf function. Shouldn't be in script for final version. 
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

# Suppose that TR=2. We know this is not a good assumption.
# Also need to look into the hrf function. 
TR = 2
tr_times = np.arange(0, 30, 2)
hrf_at_trs = hrf(tr_times)
n_vols = data.shape[-1]
events = events2neural(pathtodata+"model/model001/onsets/task001_run001/cond001.txt", TR, n_vols)
events = np.abs(events-1) # Swapping 0s and 1s. 
convolved = np.convolve(events, hrf_at_trs)
n_to_remove = len(hrf_at_trs) - 1
convolved = convolved[:-n_to_remove]

# Now get the estimated coefficients and design matrix for doing
# regression on the convolved time course. 
B, X = glm(data, convolved)

# Some diagnostics. 
MRSS, fitted, residuals = glm_diagnostics(B, X, data)

# Print out the mean MRSS.
print(np.mean(MRSS))

# Plot the time course for a single voxel with the fitted values. 
# Looks pretty bad. 
plt.plot(data[42, 32, 19])
plt.plot(fitted[42, 32, 19])
plt.savefig("glm_plot.png")
