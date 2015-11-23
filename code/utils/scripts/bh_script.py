""" Script for the Benjamini-Hochberg function.
Run with: 
    python bh_script.py
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

# Load functions
from stimuli import events2neural
from event_related_fMRI_functions import hrf_single, convolution_specialized
from Image_Visualizing import present_3d, make_mask
from glm import glm
from hypothesis import t_stat
from event_related_fMRI_functions import hrf_single, convolution_specialized
from benjamini_hochberg import bh_procedure

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

B,t,df,p = t_stat(data, my_hrf, np.array([0,1]))



#########################
# c. Benjamini-Hochberg #
#########################
print("# ======= Beginning the Benjamini-Hochberg procedure now. ======= #")
p_vals = p.T

# a fairly large false discovery rate
Q = .25
significant_pvals = bh_procedure(p_vals, Q)

# Reshape significant_pvals
reshaped_sig_p = np.reshape(significant_pvals, data.shape[:-1])
slice_reshaped_sig_p = reshaped_sig_p[...,7]
original_slice = data[...,7]

# ==== Visualization of Masked data over original brain data ==== #

# ==== No Mask ==== #
plt.imshow(slice_reshaped_sig_p)
plt.colorbar()
plt.title('Significant p-values (No mask)')
plt.savefig(location_of_images+"significant_p_slice_NOMASK.png")
plt.close()
print("Initial plot with NO MASK done.")


# ==== varying the Q value = .005 (FDR) pt 2 ==== #
Q2 = .005

significant_pvals2 = bh_procedure(p_vals, Q2)

# Reshape significant_pvals
reshaped_sig_p2 = np.reshape(significant_pvals2, data.shape[:-1])
slice_reshaped_sig_p2 = reshaped_sig_p2[...,7]

masked_data2 = make_mask(original_slice, reshaped_sig_p2, fit=False)

plt.imshow(present_3d(masked_data2))
plt.colorbar()
plt.title('Slice with Significant p-values (Q = .005)')
plt.savefig(location_of_images+"significant_p_slice2.png")
plt.close()
print("Initial plot with Q = .005 done.")

# ==== varying the Q value = .10 (FDR) pt 1 ==== #
Q1 = .10

significant_pvals1 = bh_procedure(p_vals, Q1)

# Reshape significant_pvals
reshaped_sig_p1 = np.reshape(significant_pvals1, data.shape[:-1])
slice_reshaped_sig_p1 = reshaped_sig_p1[...,7]

masked_data1 = make_mask(original_slice, reshaped_sig_p1, fit=False)

plt.imshow(present_3d(masked_data1))
plt.colorbar()
plt.title('Slice with Significant p-values (Q = .10)')
plt.savefig(location_of_images+"significant_p_slice1.png")
plt.close()
print("Initial plot with Q = .10 done.")

# ==== varying the Q value = .25 (FDR) pt 0 ==== #
masked_data = make_mask(original_slice, reshaped_sig_p, fit=False)

plt.imshow(present_3d(masked_data))
plt.colorbar()
plt.title('Slice with Significant p-values (Q = .25)')
plt.savefig(location_of_images+"significant_p_slice.png")
plt.close()
print("Initial plot with Q = .25 done.")

# ==== varying the Q value = .5 (FDR) pt 3 ==== #
Q3 = .5

significant_pvals3 = bh_procedure(p_vals, Q3)

# Reshape significant_pvals
reshaped_sig_p3 = np.reshape(significant_pvals3, data.shape[:-1])
slice_reshaped_sig_p3 = reshaped_sig_p3[...,7]

masked_data3 = make_mask(original_slice, reshaped_sig_p3, fit=False)

plt.imshow(present_3d(masked_data3))
plt.colorbar()
plt.title('Slice with Significant p-values (Q = .5)')
plt.savefig(location_of_images+"significant_p_slice3.png")
plt.close()
print("Initial plot with Q = .5 done.")