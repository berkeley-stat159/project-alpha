"""
Final script to smooth all BOLD files

Imports functions from multiple locations and see test files in correct 
	folders.

Potential variants:
	- This currently uses sigma = 1 and fwhm = (2*np.sqrt(2 *np.log(2))) * sigma 
    	whereas the paper uses 5.5mm for its fwhm
	- Saves affine from non-smooth data file.
"""

import numpy as np
import nibabel as nib
import os
import sys
import pandas as pd


# Relative path to subject all the subjects
project_path          = "../../"
path_to_data         = project_path+"data/ds009/"
location_of_images    = project_path+"images/"
location_of_functions = project_path+"code/utils/functions/" 

final_data            = "../data/"
behav_suffix          = "/behav/task001_run001/behavdata.txt"

sys.path.append(location_of_functions)


# Load smoothing function
from smooth import smoothvoxels

sub_list = os.listdir(path_to_data)
sub_list = [i for i in sub_list if 'sub' in i]


# Progress bar
toolbar_width = len(sub_list)
sys.stdout.write("Smoothing data, with sigma=1:  ")
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width + 1)) # Return to start of line, after '['



# Saving to compare number of cuts in the beginning
num_cut = np.zeros(len(sub_list))
i = 0

for name in sub_list:

	# Amount of beginning TRs not standardized at 6
	behav = pd.read_table(path_to_data + name + behav_suffix, sep = " ")
	num_TR = float(behav["NumTRs"])

	img = nib.load(path_to_data + name + "/BOLD/task001_run001/bold.nii.gz")
	
	affine = img.affine
	data = img.get_data().astype(float)


	first_n_vols = data.shape[-1]
	num_TR_cut = int(first_n_vols - num_TR)
	num_cut[i] = num_TR_cut
	i += 1



	data = data[...,num_TR_cut:] 
	

	#########################
	#  Smoothing per slice  #
	#########################

	smoothed_data = np.zeros(data.shape)
	for time in np.arange(data.shape[-1]):
		# Kind of arbitrary chosen time

		sigma = 1
		smoothed_data[..., time] = smoothvoxels(data, sigma, time)


	
	smoothed_data
	img = nib.Nifti1Image(smoothed_data, affine)
	nib.save(img,os.path.join(final_data + "smooth/", str(name) + "_bold_smoothed.nii"))
	### 266.3 MB for first one


	sys.stdout.write("-")
	sys.stdout.flush()

sys.stdout.write("\n")



