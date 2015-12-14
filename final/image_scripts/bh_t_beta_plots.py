"""
Plot producing script for BH, t-value, beta value analysis

"""

from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

project_path = "../../"
location_of_functions = project_path+"code/utils/functions/"
 
sys.path.append(location_of_functions)


from Image_Visualizing import present_3d



# mean_across for all the process outputs for each subject
final_bh = np.load("../data/bh_t_beta/final_bh_average.npy")
final_t = np.load("../data/bh_t_beta/final_t_average.npy")
final_beta = np.load("../data/bh_t_beta/final_beta_average.npy")


#####################################
# Benjamini Hochberg Plots Q = 0.25 #
#####################################

plt.imshow(present_3d(final_bh), interpolation = 'nearest', cmap = 'gray')
plt.title("Mean BH Value Across 24 Subjects with Q = .25")
plt.colorbar()
plt.savefig("../../images/bh_mean_final.png")
plt.close()

######################################
# T-statistic Plots Proportion = 0.1 #
######################################


desired_image=present_3d(final_t)
desired_image[320:,256:]=.5

plt.imshow(desired_image, interpolation = 'nearest', cmap = 'gray')
plt.title("Mean t_grouping Value Across 24 Subjects with proportion = .1")
plt.colorbar()
plt.savefig("../../images/tgroup_mean_final.png")
plt.close()


######################################
# Beta-value Plots Proportion = 0.2  #
######################################

plt.imshow(present_3d(final_beta), interpolation = 'nearest', cmap = 'gray')
plt.title("Mean beta_grouping Value Across 24 Subjects with proportion = .2")
plt.colorbar()
plt.savefig("../../images/betagroup_mean_final.png")
plt.close()


