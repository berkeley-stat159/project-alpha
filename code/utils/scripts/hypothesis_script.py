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

# Load events2neural from the stimuli module.
from stimuli import events2neural
from event_related_fMRI_functions import hrf_single, convolution_specialized

# Load our GLM functions. 
from glm import glm
from hypothesis import t_stat

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


#=================================================

""" Run hypothesis testing script"""

B,t,df,p = t_stat(data, my_hrf, np.array([0,1]))

print(t,p)
