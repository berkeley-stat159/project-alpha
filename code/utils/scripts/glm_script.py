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

#######################
# a. (my) convolution #
#######################

all_stimuli=np.array(sorted(list(cond2[:,0])+list(cond3[:,0])+list(cond1[:,0]))) # could also just x_s_array
my_hrf = convolution_specialized(all_stimuli,np.ones(len(all_stimuli)),hrf_single,np.linspace(0,239*2-2,239))


##################
# b. np.convolve #
##################

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


#############################
#############################
# Analysis and diagonistics #
#############################
#############################

#######################
# a. (my) convolution #
#######################

# Now get the estimated coefficients and design matrix for doing
# regression on the convolved time course. 
B_my, X_my = glm(data, my_hrf)

# Some diagnostics. 
MRSS_my, fitted_my, residuals_my = glm_diagnostics(B_my, X_my, data)

# Print out the mean MRSS.
print("MRSS using 'my' convolution function: "+str(np.mean(MRSS_my)))

# Plot the time course for a single voxel with the fitted values. 
# Looks pretty bad. 
plt.plot(data[41, 47, 2]) #change from cherry-picking 
plt.plot(fitted_my[41, 47, 2])
plt.savefig(location_of_images+"glm_plot_my.png")
plt.close()


##################
# b. np.convolve #
##################

# Now get the estimated coefficients and design matrix for doing
# regression on the convolved time course. 
B_np, X_np = glm(data, np_hrf)

# Some diagnostics. 
MRSS_np, fitted_np, residuals_np = glm_diagnostics(B_np, X_np, data)

# Print out the mean MRSS.
print("MRSS using np convolution function: "+str(np.mean(MRSS_np)))

# Plot the time course for a single voxel with the fitted values. 
# Looks pretty bad. 
plt.plot(data[41, 47, 2])
plt.plot(fitted_np[41, 47, 2])
plt.savefig(location_of_images+"glm_plot_np.png")
plt.close()




X_my3=np.ones((data.shape[-1],4))
for i in range(2):
     X_my3[:,i+1]=my_hrf**(i+1)
B_my3, X_my3 = glm_multiple(data, X_my3)
MRSS_my3, fitted_my3, residuals_my3 = glm_diagnostics(B_my3, X_my3, data)
print("MRSS using 'my' convolution function, 3rd degree polynomial: "+str(np.mean(MRSS_my3))+ ", but the chart looks better")

plt.plot(data[41, 47, 2])
plt.plot(fitted_my3[41, 47, 2])
plt.savefig(location_of_images+"glm_plot_my3.png")
plt.close()


