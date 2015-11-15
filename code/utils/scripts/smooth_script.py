""" Script for smooth function.
Run with: 
    python smooth_script.py

in the scripts directory
"""

import numpy as np
import itertools
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import nibabel as nib
import os
import sys

# Relative path to subject 1 data
pathtodata = "../../../data/ds009/sub001/"
condition_location=pathtodata+"model/model001/onsets/task001_run001/"
location_of_images="../../../images/"

#sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
sys.path.append("../functions")

# Load events2neural from the stimuli module.
#from stimuli import events2neural
#from event_related_fMRI_functions import hrf_single, convolution_specialized

# Load our GLM functions. 
#from glm import glm, glm_diagnostics, glm_multiple

# Load smoothing function
from smooth import smoothvoxels
from Image_Visualizing import present_3d

# Load the image data for subject 1.
img = nib.load(pathtodata+"BOLD/task001_run001/bold.nii.gz")
data = img.get_data()
data = data[...,6:] # Knock off the first 6 observations.

#######################
# a. (my) smoothing   #
#######################

# Kind of arbitrary chosen time
time = 7
original_slice = data[..., 7]
# full width at half maximum (FWHM) 
fwhm = 1.5
smoothed_slice = smoothvoxels(data, fwhm, time)

# visually compare original_slice to smoothed_slice
plt.imshow(present_3d(smoothed_slice))
plt.colorbar()
plt.title('Smoothed Slice')
plt.clim(0,1600)
plt.savefig(location_of_images+"smoothed_slice.png")

plt.close()

plt.imshow(present_3d(original_slice))
plt.colorbar()
plt.title('Original Slice')
plt.clim(0,1600)
plt.savefig(location_of_images+"original_slice.png")

plt.close()

