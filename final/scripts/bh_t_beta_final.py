"""
Final script for BH, t-value, beta value analysis

For each subject: collect the p-values, t-values, beta-values. 
	Compute the bh_procedure, t_grouping, beta grouping (similar to t_grouping idea)
	Average the 3d outputs (mean_across by Hiro?)

"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import sys
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
import pandas as pd
from scipy.stats import t as t_dist

project_path          = "../../"
path_to_data          = project_path+"data/ds009/sub001"
location_of_images    = project_path+"images/"
location_of_functions = project_path+"code/utils/functions/" 
final_data            = "../data/"
behav_suffix           = "/behav/task001_run001/behavdata.txt"
smooth_data           =  final_data + 'smooth/'
hrf_data              = final_data + 'hrf/'

sys.path.append(location_of_functions)

# list of subjects
sub_list = os.listdir(path_to_data)[1:]

from tgrouping import t_grouping_neighbor
from mask_phase_2_dimension_change import masking_reshape_start, masking_reshape_end
from mask_phase_2_dimension_change import neighbor_smoothing, neighbor_smoothing_binary

from Image_Visualizing import present_3d, make_mask
from benjamini_hochberg import bh_procedure

#p_3d = np.load("../data/p-values/sub001_pvalue.npy")
#t_3d = np.load("../data/t_stat/sub001_tstat.npy")
#beta_3d = np.load("../data/betas/sub001_beta.npy")

mask = nib.load(path_to_data + '/anatomy/inplane001_brain_mask.nii.gz')
mask_data = mask.get_data()
rachels_ones = np.ones((64, 64, 34))
fitted_mask = make_mask(rachels_ones, mask_data, fit = True)
fitted_mask[fitted_mask > 0] = 1

#####################################
# Run bh_procedure for each subject #
#####################################

# find BH output for each subject and save them all
q = .2
neighbors = 1
#sub_bh = []

#Create empty array for the p-values per voxel per subject
bh_mean = np.zeros((64, 64, 34, 24))

for i in sub_list:
	p_3d = np.load("../data/p-values/" + i + "_pvalue.npy")
	p_1d = np.ravel(p_3d)

	mask = fitted_mask
	mask_1d = np.ravel(mask)
	p_bh = p_1d[mask_1d == 1]

	bh_first = bh_procedure(p_bh, q)
	bh_3d    = masking_reshape_end(bh_first, mask, off_value = .5)
	bh_3d[bh_3d < .5] = 0
	bh_3d_1_good = 1 - bh_3d

	bh_final  = neighbor_smoothing_binary(bh_3d_1_good, neighbors)

	bh_mean[..., int(i[-1])] = bh_final
	#sub_bh.append(bh_final)

# mean_across for all the BH outputs for each subject
final_bh = present_3d(np.mean(bh_mean, axis = 3))

# plot/save the result
plt.imshow(final_bh, interpolation = 'nearest', cmap = 'seismic')
plt.title("Mean BH Value Across 25 Subjects with Q = .2")

zero_out = max(abs(np.min(final_bh)), np.max(final_bh))
plt.clim(-zero_out, zero_out)
plt.colorbar()
#plt.savefig("../../../paper/images/bh_mean_final.png")
plt.close()

# (from parameter_selection_final plotting)
#plt.contour(present_bh, interpolation = "nearest", colors = "k", alpha = 1)
#plt.imshow(behind,interpolation="nearest",cmap="seismic")
#plt.title("Benjamini Hochberg on slice 15 and contours \n (with varying Q and # neighbors)")
#x=32+64*np.arange(5)
#labels = neighbors1
#plt.xticks(x, labels)
#plt.xlabel("Number of Neighbors")
#labels2 = q1
#y=32+64*np.arange(6)
#plt.yticks(y, labels2)
#plt.ylabel("Q")
#plt.colorbar()
#plt.savefig(location_of_images+"bh_compare_15_plus_contours.png")
#plt.close()

#####################################
# Run t_grouping for each subject   #
#####################################
prop = .1
#sub_tgroup = []
t_mean = np.zeros((64, 64, 34, 24))

for i in sub_list:
	t_3d = np.load("../data/t_stat/" + i + "_tstat.npy")

	mask = fitted_mask
	t_group = t_grouping_neighbor(t_3d, mask, prop, neighbors = neighbors,
					prop = True, abs_on = True, binary = True, off_value = 0, masked_value = .5)[0]

	t_mean[..., int(i[-1])] = t_group

# mean_across for all the t_grouping outputs for each subject
final_t = present_3d(np.mean(t_mean, axis = 3))

# plot/save the result
plt.imshow(final_t, interpolation = 'nearest', cmap = 'seismic')
plt.title("Mean t_grouping Value Across 25 Subjects with proportion = .1")

zero_out = max(abs(np.min(final_t)), np.max(final_t))
plt.clim(-zero_out, zero_out)
plt.colorbar()
#plt.savefig("../../../paper/images/tgroup_mean_final.png")
plt.close()

######################################
# Run beta grouping for each subject #
######################################
prop_beta = .15

beta_mean = np.zeros((64, 64, 34, 24))

for i in sub_list:
	beta_3d = np.load("../data/betas/" + i + "_beta.npy")
	beta_group = t_grouping_neighbor(beta_3d, mask, prop_beta, neighbors = neighbors,
						prop = True, abs_on = True, binary = True, off_value = 0, masked_value = .5)[0]

	beta_mean[..., int(i[-1])] = beta_group

# mean_across for all the t_grouping outputs for each subject
final_beta = present_3d(np.mean(beta_mean, axis = 3))

# plot/save the result
plt.imshow(final_beta, interpolation = 'nearest', cmap = 'seismic')
plt.title("Mean beta_grouping Value Across 25 Subjects with proportion = .15")

zero_out = max(abs(np.min(final_beta)), np.max(final_beta))
plt.clim(-zero_out, zero_out)
plt.colorbar()
#plt.savefig("../../../paper/images/betagroup_mean_final.png")
plt.close()