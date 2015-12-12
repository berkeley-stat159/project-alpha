import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools
import nibabel as nib
import os

name = input("Subject name (like 'sub001'): ")

project_path          = "../../"
path_to_data          = project_path+"data/ds009/"+name
location_of_images    = project_path+"images/"
location_of_functions = project_path+"code/utils/functions/" 
final_data            = "../data/"
behav_suffix           = "/behav/task001_run001/behavdata.txt"
smooth_data           =  final_data + 'smooth/'
hrf_data              = final_data + 'hrf/'


sys.path.append(location_of_functions)


from tgrouping import t_grouping_neighbor
from mask_phase_2_dimension_change import masking_reshape_start, masking_reshape_end, neighbor_smoothing,neighbor_smoothing_binary
from Image_Visualizing import present_3d, make_mask
from benjamini_hochberg import bh_procedure


p_3d = np.load("../data/p-values/"+name+"_pvalue_fourier.npy")
t_3d = np.load("../data/t_stat/"+name+"_tstat.npy")
beta_3d = np.load("../data/betas/"+name+"_beta.npy")


mask = nib.load(path_to_data + '/anatomy/inplane001_brain_mask.nii.gz')
mask_data = mask.get_data()
rachels_ones = np.ones((64, 64, 34))
fitted_mask = make_mask(rachels_ones, mask_data, fit = True)
fitted_mask[fitted_mask>0]=1



plt.imshow(present_3d(beta_3d*fitted_mask),cmap="seismic")
plt.colorbar()
plt.clim(-np.max(abs(beta_3d)),np.max(abs(beta_3d)))
plt.title(name+" beta values")
plt.yticks([])
plt.xticks([])
plt.savefig(location_of_images+name+"_"+"beta.png")
plt.close()


plt.imshow(present_3d(t_3d*fitted_mask),cmap="seismic")
plt.colorbar()
plt.clim(-np.max(abs(t_3d)),np.max(abs(t_3d)))
plt.title(name+" t values")
plt.yticks([])
plt.xticks([])
plt.savefig(location_of_images+name+"_"+"t.png")
plt.close()


