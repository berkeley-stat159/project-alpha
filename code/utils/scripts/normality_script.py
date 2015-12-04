# Loading modules.
import numpy as np
import nibabel as nib
from scipy.stats import shapiro
import os
import sys

# Relative path to subject 1 data
pathtodata = "../../../data/ds009/sub001/"
condition_location=pathtodata+"model/model001/onsets/task001_run001/"
location_of_images="../../../images/"

sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))

# Load events2neural from the stimuli module.
from stimuli import events2neural
from event_related_fMRI_functions import hrf_single, convolution_specialized

# Load our GLM functions. 
from glm import glm, glm_diagnostics

# Load our normality functions. 
from normality import check_sw, check_sw_masked

# Load the image data for subject 1.
img = nib.load(pathtodata+"BOLD/task001_run001/bold.nii.gz")
data = img.get_data()
data = data[...,6:] # Knock off the first 6 observations.

cond1=np.loadtxt(condition_location+"cond001.txt")
cond2=np.loadtxt(condition_location+"cond002.txt")
cond3=np.loadtxt(condition_location+"cond003.txt")


#################
#np.convolve
################

# initial needed values
TR = 2
tr_times = np.arange(0, 30, TR)
hrf_at_trs = np.array([hrf_single(x) for x in tr_times])
n_vols=data.shape[-1]

# creating the .txt file for the events2neural function
cond_all=np.row_stack((cond1,cond2,cond3))
cond_all=sorted(cond_all,key= lambda x:x[0])
np.savetxt(condition_location+"cond_all.txt",cond_all)

neural_prediction = events2neural(condition_location+"cond_all.txt",TR,n_vols)
convolved = np.convolve(neural_prediction, hrf_at_trs) # hrf_at_trs sample data
N = len(neural_prediction)  # N == n_vols == 173
M = len(hrf_at_trs)  # M == 12
np_hrf=convolved[:N]

###################
# From GLM function
###################

np_B, np_X = glm(data, np_hrf)


####################################
# GLM Diagnostics (to get residuals)
###################################

np_MRSS, np_fitted, np_residuals = glm_diagnostics(np_B, np_X, data)

###########################
#Shapiro-Wilks on Residuals
###########################
# Shapiro-Wilks: tests the null hypothesis that the data was 
# drawn from a normal distribution.

# Using 4-d residuals.
sw_pvals = check_sw(np_residuals)
print(np.mean(sw_pvals > 0.05))

# Using 2-d residuals (voxel by time), which is kind of silly now, 
# but will be important when running models on masked data. 
resid_2d = np_residuals.reshape((-1,np_residuals.shape[-1]))
sw_pvals_2d = check_sw_masked(resid_2d)
print(np.mean(sw_pvals_2d > 0.05))
print("Should be the same.")

