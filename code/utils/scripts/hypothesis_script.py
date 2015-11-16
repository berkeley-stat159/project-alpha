""" Script for Hypothesis testing functions.
Run with: 
    python hypothesis_script.py
"""

# Loading modules.
import os
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
import nibabel as nib
import sys
import numpy.linalg as npl

# Paths. Use your own. 

pathtodata = "../../../data/ds009/sub001/"
condition_location=pathtodata+"model/model001/onsets/task001_run001/"
location_of_images="../../../images/"

sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))

# Load events2neural from the stimuli module
from stimuli import events2neural
from event_related_fMRI_functions import hrf_single, convolution_specialized

# Load our GLM functions. 
from glm import glm
from hypothesis import t_stat

# Load our convolution and hrf function
from event_related_fMRI_functions import hrf_single, convolution_specialized

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

# Suppose that TR=2. We know this is not a good assumption.
# Also need to look into the hrf function. 
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







#######################
# a. (my) convolution #
#######################

all_stimuli=np.array(sorted(list(cond2[:,0])+list(cond3[:,0])+list(cond1[:,0]))) # could also just x_s_array
my_hrf = convolution_specialized(all_stimuli,np.ones(len(all_stimuli)),hrf_single,np.linspace(0,239*2-2,239))


#=================================================

""" Run hypothesis testing script"""

B_my,t_my,df,p_my = t_stat(data, my_hrf, np.array([0,1]))

print("'my' convolution single regression (t,p):")
print(t_my,p_my)
print("means of (t,p) for 'my' convolution: (" +str(np.mean(t_my))+str(np.mean(p_my)) +")")

B_np,t_np,df,p_np = t_stat(data, np_hrf, np.array([0,1]))

print("np convolution single regression (t,p):")
print(t_np,p_np)
print("means of (t,p) for np convolution: (" +str(np.mean(t_np))+str(np.mean(p_np)) +")")
B,t,df,p = t_stat(data, my_hrf, np.array([0,1]))

