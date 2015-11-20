""" Script for smooth function.
Run with: 
    python smooth_script_test.py

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

project_path          = "../../"
path_to_data          = project_path+"data/ds009/"
location_of_images    = project_path+"images/"
location_of_functions = project_path+"code/utils/functions/" 
final_data            = "../data/"

#sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
sys.path.append(location_of_functions)

# Load events2neural from the stimuli module.
#from stimuli import events2neural
#from event_related_fMRI_functions import hrf_single, convolution_specialized

# Load our GLM functions. 
#from glm import glm, glm_diagnostics, glm_multiple


# Load smoothing function
from smooth import smoothvoxels
from Image_Visualizing import present_3d

sub_list = os.listdir(path_to_data)[1:]


# Progress bar
toolbar_width=len(sub_list)
sys.stdout.write("Smoothing data, with 'fwhm = 1.5':  ")
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['



for name in sub_list:
	img = nib.load(path_to_data+ name+ "/BOLD/task001_run001/bold.nii.gz")
	
	affine=img.affine 

	data = img.get_data()
	data = data[...,6:] 
	

	#########################
	#  smoothing per slice  #
	#########################

	smoothed_data =np.zeros(data.shape)
	for time in np.arange(data.shape[-1]):
		# Kind of arbitrary chosen time
		fwhm = 1.5
		smoothed_data[...,time]= smoothvoxels(data, fwhm, time)


	
	smoothed_data
	img = nib.Nifti1Image(smoothed_data, affine)
	nib.save(img,os.path.join(final_data + "smooth/",str(name)+"_bold_smoothed.nii"))
	### 266.3 MB for first one


	sys.stdout.write("-")
	sys.stdout.flush()

sys.stdout.write("\n")


# if we want to save them all:
# cost of function:  2.5 s * 24  (1 minute to smooth)
# cost of saving ~ 4 minutes




