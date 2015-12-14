""" Script for the Benjamini-Hochberg function on subject001.
Run with: 
    python bh_script.py
"""

# Loading modules.
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import sys
import numpy.linalg as npl

# Relative paths
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

# Load the image data for subject 1.
img = nib.load(pathtodata+"BOLD/task001_run001/bold.nii.gz")
data = img.get_data()
data = data[...,6:] # Knock off the first 6 observations.

#Load convolution files
cond1=np.loadtxt(condition_location+"cond001.txt")
cond2=np.loadtxt(condition_location+"cond002.txt")
cond3=np.loadtxt(condition_location+"cond003.txt")


#Convolution and t-values
all_stimuli=np.array(sorted(list(cond2[:,0])+list(cond3[:,0])+list(cond1[:,0])))
my_hrf = convolution_specialized(all_stimuli,np.ones(len(all_stimuli)),hrf_single,np.linspace(0,239*2-2,239))

B,t,df,p = t_stat(data, my_hrf, np.array([0,1]))



#########################
# Benjamini-Hochberg #
#########################


print("# ==== BEGIN Visualization of Masked data over original brain data ==== #")

p_vals = p.T # shape of p_vals is (139264, 1)

print("# ==== No Mask, bh_procedure ==== #")
# a fairly large false discovery rate
Q = .4
significant_pvals = bh_procedure(p_vals, Q)

# Reshape significant_pvals to shape of data
reshaped_sig_p = np.reshape(significant_pvals, data.shape[:-1])
slice_reshaped_sig_p = reshaped_sig_p[...,7]
original_slice = data[...,7]

plt.imshow(slice_reshaped_sig_p)
plt.colorbar()
plt.title('Significant p-values (No mask)')
plt.savefig(location_of_images+"NOMASK_significant_p_slice.png")
plt.close()
print("# ==== END No Mask, bh_procedure ==== #")



print("# ==== BEGIN varying the Q value = .005 (FDR) ==== #")
Q = .005

significant_pvals = bh_procedure(p_vals, Q)

# Reshape significant_pvals
reshaped_sig_p = np.reshape(significant_pvals, data.shape[:-1])
slice_reshaped_sig_p = reshaped_sig_p[...,7]

masked_data = make_mask(original_slice, reshaped_sig_p, fit=False)

plt.imshow(present_3d(masked_data))
plt.clim(0, 1600)
plt.colorbar()
plt.title('Slice with Significant p-values (Q = .005)')
plt.savefig(location_of_images+"significant_p_slice1.png")
plt.close()
print("# ==== END plot with Q = .005 done. ==== #")


print("# ==== BEGIN varying the Q value = .05 (FDR) ==== #")
Q = .05

significant_pvals = bh_procedure(p_vals, Q)

# Reshape significant_pvals
reshaped_sig_p = np.reshape(significant_pvals, data.shape[:-1])
slice_reshaped_sig_p = reshaped_sig_p[...,7]

masked_data = make_mask(original_slice, reshaped_sig_p, fit=False)

plt.imshow(present_3d(masked_data))
plt.colorbar()
plt.title('Slice with Significant p-values (Q = .05)')
plt.savefig(location_of_images+"significant_p_slice2.png")
plt.close()
print("# ==== END plot with Q = .05 done. ==== #")


print("# ==== BEGIN varying the Q value = .10 (FDR) ==== #")
Q = .10

significant_pvals = bh_procedure(p_vals, Q)

# Reshape significant_pvals
reshaped_sig_p = np.reshape(significant_pvals, data.shape[:-1])
slice_reshaped_sig_p = reshaped_sig_p[...,7]

masked_data = make_mask(original_slice, reshaped_sig_p, fit=False)

plt.imshow(present_3d(masked_data))
plt.colorbar()
plt.title('Slice with Significant p-values (Q = .10)')
plt.savefig(location_of_images+"significant_p_slice3.png")
plt.close()
print("# ==== END plot with Q = .10 done. ==== #")

print("# ==== BEGIN the Q value = .25 (FDR) ==== #")
Q = .25
significant_pvals = bh_procedure(p_vals, Q)
slice_reshaped_sig_p = reshaped_sig_p[...,7]

masked_data = make_mask(original_slice, reshaped_sig_p, fit=False)

plt.imshow(present_3d(masked_data))
plt.colorbar()
plt.title('Slice with Significant p-values (Q = .25)')
plt.savefig(location_of_images+"significant_p_slice4.png")
plt.close()
print("# ==== END plot with Q = .25 done. ==== #")

print("# ==== BEGIN the Q value = .5 (FDR) ==== #")
Q = .5

significant_pvals = bh_procedure(p_vals, Q)

# Reshape significant_pvals
reshaped_sig_p = np.reshape(significant_pvals, data.shape[:-1])
slice_reshaped_sig_p = reshaped_sig_p[...,7]

masked_data = make_mask(original_slice, reshaped_sig_p, fit=False)

plt.imshow(present_3d(masked_data))
plt.colorbar()
plt.title('Slice with Significant p-values (Q = .5)')
plt.savefig(location_of_images+"significant_p_slice5.png")
plt.close()
print("# ==== END plot with Q = .5 done. === #")


print("# ==== BEGIN mask_phase_2_dimension_change plotting (LOL) ==== #")
Q = .45
mask = nib.load(pathtodata + '/anatomy/inplane001_brain_mask.nii.gz')
mask_data = mask.get_data()

rachels_ones = np.ones(data.shape[:-1])
fitted_mask = make_mask(rachels_ones, mask_data, fit = True)
fitted_mask[fitted_mask > 0] = 1

mask_new = mask_data[::2, ::2, :]
assert(mask_new.shape == fitted_mask.shape)

p_vals_3d = p_vals.reshape(data.shape[:-1])

# call masking_reshape_start function, reshaped to 1d output
to_1d = masking_reshape_start(p_vals_3d, mask_new)

# call bh_procedure where Q = .45
bh_1d=bh_procedure(to_1d, Q)

# call masking_reshape_end function, to reshape back into 3d shape
to_3d = 2*masking_reshape_end(bh_1d, mask_new, .5) - 1

# plot this
plt.imshow(present_3d(to_3d),interpolation='nearest', cmap='seismic')
plt.title(str(Q) + " with masked p-values")

plt.savefig(location_of_images + "masked_pval_" + str(Q) + ".png")
plt.figure()
plt.close()
print("# ==== END mask_phase_2_dimension_change plotting ==== #")

# Smoothing/clustering on the masked p-values
print("# ==== BEGIN smoothed masked p-value plotting ==== #")

# Where the neighbor_smoothing function should start
print("# == NEIGHBOR SMOOTHING 5")
smoothed = neighbor_smoothing(to_3d, 5)

plt.imshow(present_3d(smoothed), interpolation='nearest', cmap='seismic')
plt.title(str(Q) + " with neightbor-smoothing factor 5")

plt.savefig(location_of_images + "5neighbor_smooth_pval_" + str(Q) + ".png")
plt.figure()
plt.close()

print("# == NEIGHBOR SMOOTHING 10")
smoothed = neighbor_smoothing(to_3d, 10)
plt.imshow(present_3d(smoothed), interpolation='nearest', cmap='seismic')
plt.title(str(Q) + " with neightbor-smoothing factor 10")
plt.savefig(location_of_images + "10neighbor_smooth_pval_" + str(Q) + ".png")
plt.figure()
plt.close()

print("# ==== END smoothed masked p-value plotting ==== #")


