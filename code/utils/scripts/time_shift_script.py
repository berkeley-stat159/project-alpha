"""
Script to do time shift on convolved time course for a single
subject. 
"""

# Loading modules.
from __future__ import absolute_import, division, print_function
import numpy as np
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
from event_related_fMRI_functions import hrf_single
from time_shift import time_shift

# Load the image data for subject 1.
img = nib.load(pathtodata+"BOLD/task001_run001/bold.nii.gz")
data = img.get_data()
data = data[...,6:] # Knock off the first 6 observations.

# Load the three conditions. 
cond1=np.loadtxt(condition_location+"cond001.txt")
cond2=np.loadtxt(condition_location+"cond002.txt")
cond3=np.loadtxt(condition_location+"cond003.txt")

# Initialize needed values
TR = 2
tr_times = np.arange(0, 30, TR)
hrf_at_trs = np.array([hrf_single(x) for x in tr_times])
n_vols=data.shape[-1]

# Creating the .txt file for the events2neural function
cond_all=np.row_stack((cond1,cond2,cond3))
cond_all=sorted(cond_all,key= lambda x:x[0])
np.savetxt(condition_location+"cond_all.txt",cond_all)

# Event stimuli
neural_prediction = events2neural(condition_location+"cond_all.txt",TR,n_vols)

# Convolved time course using np.convolve
convolved = np.convolve(neural_prediction, hrf_at_trs) 

# Get the back-shifted time course. 
convolved2, shifted = time_shift(convolved, neural_prediction, TR)

# Compare before and after shifting.
plt.plot(neural_prediction)
plt.plot(convolved2)
plt.plot(shifted)
plt.savefig(location_of_images + "shifted.png")

print("Shifted time course more closely matches stimuli.")







#######################
# Ben's improvements: #
#######################

# using the cond_all from above
cond_all=np.array(cond_all)[:,0]

from time_shift import make_shift_matrix,time_correct

delta_y=2*(np.arange(34))/34


shifted=make_shift_matrix(cond_all,delta_y)
plt.close()
for i in range(delta_y.shape[0]):
    plt.plot(cond_all,shifted[:,i])
plt.xlim(0,10)
plt.ylim(0,10)
plt.savefig(location_of_images + "ben_stupid_linear_shift.png")
plt.close()


# second

from event_related_fMRI_functions import hrf_single, np_convolve_30_cuts


def make_convolve_lambda(hrf_function,TR,num_TRs):
    convolve_lambda=lambda x: np_convolve_30_cuts(x,np.ones(x.shape[0]),hrf_function,TR,np.linspace(0,(num_TRs-1)*TR,num_TRs),15)[0]
    return convolve_lambda

convolve_lambda=make_convolve_lambda(hrf_single,2,239)

hrf_matrix=time_correct(convolve_lambda,shifted,239)


TR=2
num_TRs=239
for i in [0,1,10,33]:
    plt.plot(np.linspace(0,(num_TRs-1)*TR,num_TRs),hrf_matrix[:,i])

plt.xlim(0,50)
plt.ylim(-.5,2)

plt.savefig(location_of_images + "hrf_time_correction.png")
plt.close()

