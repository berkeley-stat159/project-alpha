"""
Final script for BH, t-value, beta value analysis

For each subject: collect the p-values, t-values, beta-values. 
	Compute the bh_procedure, t_grouping, beta grouping (similar to t_grouping idea)
	Average the 3d outputs (mean_across)

"""

from __future__ import absolute_import, division, print_function
import numpy as np
import nibabel as nib
import os
import sys
import pandas as pd
from scipy.stats import t as t_dist

project_path          = "../../"
path_to_data          = project_path + "data/ds009/"
location_of_images    = project_path + "images/"
location_of_functions = project_path + "code/utils/functions/" 
final_data            = "../data/"
behav_suffix           = "/behav/task001_run001/behavdata.txt"
smooth_data           =  final_data + 'smooth/'
hrf_data              = final_data + 'hrf/'

sys.path.append(location_of_functions)

# list of subjects
sub_list = os.listdir(path_to_data)
sub_list = [i for i in sub_list if 'sub' in i]

# Progress bar
toolbar_width = len(sub_list)
sys.stdout.write("Clustering (BH, t, beta):  ")
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width + 1)) # return to start of line, after '['

from tgrouping import t_grouping_neighbor
from mask_phase_2_dimension_change import masking_reshape_start, masking_reshape_end, neighbor_smoothing, neighbor_smoothing_binary
from Image_Visualizing import present_3d, make_mask
from benjamini_hochberg import bh_procedure


# Giant for loop!!!
bh_mean = np.zeros((64, 64, 34, 24))
t_mean = np.zeros((64, 64, 34, 24))
beta_mean = np.zeros((64, 64, 34, 24))

neighbors = 1
q = .15
prop_t = .15
prop_beta = .15

# assign subjects a number

for i, name in enumerate(sub_list):

	# the mask for each subject
	path_to_data = project_path + "data/ds009/" + name
	mask = nib.load(path_to_data + '/anatomy/inplane001_brain_mask.nii.gz')
	mask_data = mask.get_data()
	rachels_ones = np.ones((64, 64, 34))
	fitted_mask = make_mask(rachels_ones, mask_data, fit = True)
	fitted_mask[fitted_mask > 0] = 1

	#####################################
	# Run bh_procedure for each subject #
	#####################################
	p_3d = np.load("../data/p-values/" + name + "_pvalue.npy")
	p_1d = np.ravel(p_3d)

	mask = fitted_mask
	mask_1d = np.ravel(mask)
	p_bh = p_1d[mask_1d == 1]

	bh_first = bh_procedure(p_bh, q)
	bh_3d    = masking_reshape_end(bh_first, mask, off_value = .5)
	bh_3d[bh_3d < .5] = 0
	bh_3d_1_good = 1 - bh_3d

	bh_final  = neighbor_smoothing_binary(bh_3d_1_good, neighbors)

	bh_mean[..., i] = bh_3d_1_good

	#####################################
	# Run t_grouping for each subject   #
	#####################################
	t_3d = np.load("../data/t_stat/" + name + "_tstat.npy")

	#mask = fitted_mask
	t_group = t_grouping_neighbor(t_3d, mask, prop_t, neighbors = neighbors,
					prop = True, abs_on = True, binary = True, off_value = 0, masked_value = .5)[0]

	t_mean[..., i] = t_group

	######################################
	# Run beta grouping for each subject #
	######################################
	beta_3d = np.load("../data/betas/" + name + "_beta.npy")
	beta_group = t_grouping_neighbor(beta_3d, mask, prop_beta, neighbors = neighbors,
						prop = True, abs_on = True, binary = True, off_value = 0, masked_value = .5)[0]

	beta_mean[..., i] = beta_group

	sys.stdout.write("-")
	sys.stdout.flush()

sys.stdout.write("\n")


# mean_across for all the process outputs for each subject
final_bh = np.mean(bh_mean, axis = 3)
np.save("../data/bh_t_beta/bh_all.npy",bh_mean)
np.save("../data/bh_t_beta/final_bh_average.npy", final_bh)

final_t = np.mean(t_mean, axis = 3)
np.save("../data/bh_t_beta/final_t_average.npy", final_t)
np.save("../data/bh_t_beta/t_all.npy",t_mean)


final_beta = np.mean(beta_mean, axis = 3)
np.save("../data/bh_t_beta/final_beta_average.npy", final_beta)
np.save("../data/bh_t_beta/beta_all.npy",beta_mean)

