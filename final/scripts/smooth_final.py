"""
Script to smooth all bold files, 

Imports functions from multiple locations, see test files in correct folders

Potential variants:
	- this currently uses sigma =1/ fwhm =3.5
	- saves affine from non-smooth data file
"""

import numpy as np
import itertools
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import nibabel as nib
import os
import sys
import pandas as pd


# Relative path to subject all the subjects
project_path          = "../../"
path_to_data          = project_path+"data/ds009/"
location_of_images    = project_path+"images/"
location_of_functions = project_path+"code/utils/functions/" 
final_data            = "../data/"
behav_suffix           = "/behav/task001_run001/behavdata.txt"

sys.path.append(location_of_functions)

# Load smoothing function
from smooth import smoothvoxels
from Image_Visualizing import present_3d

sub_list = os.listdir(path_to_data)[1:]

# Progress bar
toolbar_width=len(sub_list)
sys.stdout.write("Smoothing data, with sigma=1:  ")
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['



# saving to compare number of cuts in the beginning
num_cut=np.zeros(len(sub_list))
i=0

for name in sub_list:

	# amount of beginning TRs not standardized at 6
	behav=pd.read_table(path_to_data+name+behav_suffix,sep=" ")
	num_TR = float(behav["NumTRs"])

	img = nib.load(path_to_data+ name+ "/BOLD/task001_run001/bold.nii.gz")
	
	affine=img.affine # why is it 4d? 
	data = img.get_data().astype(float)


	first_n_vols=data.shape[-1]
	num_TR_cut=int(first_n_vols-num_TR)
	num_cut[i]=num_TR_cut
	i+=1



	data = data[...,num_TR_cut:] 
	

	#########################
	#  smoothing per slice  #
	#########################

	smoothed_data =np.zeros(data.shape)
	for time in np.arange(data.shape[-1]):
		# Kind of arbitrary chosen time

		sigma = 1
		fwhm = (2*np.sqrt(2 *np.log(2))) * sigma
		smoothed_data[...,time]= smoothvoxels(data, sigma, time)


	
	smoothed_data
	img = nib.Nifti1Image(smoothed_data, affine)
	nib.save(img,os.path.join(final_data + "smooth/",str(name)+"_bold_smoothed.nii"))
	### 266.3 MB for first one


	sys.stdout.write("-")
	sys.stdout.flush()

sys.stdout.write("\n")



# if we want to save them all:
# cost of function:  2.5 s * 24  (1 minute to smooth)
# cost of saving:
#1 loops, best of 3: 1min 24s per loop




