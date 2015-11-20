# Loading modules.
import numpy as np
from scipy.stats import gamma
from scipy.stats import mstats
import matplotlib.pyplot as plt
import nibabel as nib
import numpy.linalg as npl
import scipy.stats
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
from glm import glm, glm_diagnostics, glm_multiple

# Load the image data for subject 1.
img = nib.load(pathtodata+"BOLD/task001_run001/bold.nii.gz")
data = img.get_data()
data = data[...,6:] # Knock off the first 6 observations.

cond1=np.loadtxt(condition_location+"cond001.txt")
cond2=np.loadtxt(condition_location+"cond002.txt")
cond3=np.loadtxt(condition_location+"cond003.txt")


data.shape #shape of data should be (64, 64, 34, 239)


#################
#np.convolve
################

#1D convolved time course
TR = 2
tr_times = np.arange(0, 30, TR)
hrf_at_trs = np.array([hrf_single(x) for x in tr_times])
n_vols = data.shape[-1]

neural_prediction = events2neural(condition_location+"cond001.txt",TR,n_vols)
convolved = np.convolve(neural_prediction, hrf_at_trs) # hrf_at_trs sample data
N = len(neural_prediction)  # N == n_vols == 239
M = len(hrf_at_trs)  # M == 15
np_hrf=convolved[:N]

##############
# GLM function
#############

np_B, np_X = glm(data, np_hrf)


####################################
# GLM Diagnostics (to get residuals)
###################################

np_MRSS, np_fitted, np_residuals = glm_diagnostics(np_B, np_X, data)


###########################
#Shapiro-Wilks on Residuals
###########################

np_residuals.shape #should be (64, 64, 34, 239)

for i in range(64):
    for j in range(64):
        for k in range(34):
            #Shapiro-Wilks: tests the null hypothesis that the data was 
            #drawn from a normal distribution.
            sw = scipy.stats.shapiro(np_residuals[i,j,k,:])   

print "Shapiro-Wilks p-value:", sw[1] #0.0139
