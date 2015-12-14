""" Script for the tgrouping function.
Run with: 
    python tgrouping_script.py
"""
# Loading modules.
from __future__ import absolute_import, division, print_function
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
sys.path.append(os.path.join(os.path.dirname('__file__'), "../functions/"))

# Load functions
from stimuli import events2neural
from event_related_fMRI_functions import hrf_single, convolution_specialized
from Image_Visualizing import present_3d, make_mask
from glm import glm
from hypothesis import t_stat
from event_related_fMRI_functions import hrf_single, convolution_specialized
from benjamini_hochberg import bh_procedure
from mask_phase_2_dimension_change import masking_reshape_start, masking_reshape_end, neighbor_smoothing
from tgrouping import t_binary_grouping, t_grouping_neighbor

# Load the image data for subject 1.
img = nib.load(pathtodata+"BOLD/task001_run001/bold.nii.gz")
data = img.get_data()
data = data[...,6:] # Knock off the first 6 observations.

cond1=np.loadtxt(condition_location+"cond001.txt")
cond2=np.loadtxt(condition_location+"cond002.txt")
cond3=np.loadtxt(condition_location+"cond003.txt")

#######################
# convolution         #
#######################

all_stimuli=np.array(sorted(list(cond2[:,0])+list(cond3[:,0])+list(cond1[:,0]))) # could also just x_s_array
my_hrf = convolution_specialized(all_stimuli,np.ones(len(all_stimuli)),hrf_single,np.linspace(0,239*2-2,239))

B,t,df,p = t_stat(data, my_hrf, np.array([0,1]))

###############
# tgrouping   #
###############
mask = nib.load(pathtodata + '/anatomy/inplane001_brain_mask.nii.gz')
mask = mask.get_data()
inner_ones=np.ones(data.shape[:-1])
mask= make_mask(inner_ones,mask,True)

mask[mask>0]=1


t_vals=t


t_vals_3d=t_vals.reshape(data.shape[:-1])

pro=[.25,.1,.1,.05,.025]
folks=[1,1,5,5,10]

plt.close()
for i in np.arange(5):
	start,cutoff=t_grouping_neighbor(t_vals_3d,mask,pro[i],prop=True,neighbors= folks[i],abs_on=True)
	plt.imshow(present_3d(2*start-1),interpolation='nearest',cmap="seismic")
	plt.title("T statistics " +str(pro[i])+" proportion \n (cutoff=" + str(cutoff)+") , neighbors: " + str(folks[i]))
	plt.colorbar()
	plt.savefig(location_of_images+str(pro[i])+"_" + str(folks[i])+"_t.png")
	plt.close()


##################
# Beta           #
##################
b1 = B[1]
#cutoff = .6
b1_vals_3d=b1.reshape(data.shape[:-1])
pro=[.25,.1,.1,.05,.025]
folks=[1,1,5,5,10]

#plt.close()
for i in np.arange(5):
	plt.figure()
	start,cutoff=t_grouping_neighbor(b1_vals_3d,mask,pro[i],prop=True,neighbors= folks[i],abs_on=True)
	plt.imshow(present_3d(2*start-1),interpolation='nearest',cmap="seismic")
	plt.title("Beta values " +str(pro[i])+" proportion \n (cutoff=" + str(cutoff)+"), neighbors: " + str(folks[i]))
	plt.colorbar()
	plt.savefig(location_of_images+str(pro[i])+"_" + str(folks[i])+"_b.png")
	plt.close()





