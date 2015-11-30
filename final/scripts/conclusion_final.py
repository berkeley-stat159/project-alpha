

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import sys
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
import pandas as pd
from scipy.stats import t as t_dist

#from glm import glm, glm_diagnostics

project_path          = "../../"
path_to_data          = project_path+"data/ds009/"
location_of_images    = project_path+"images/"
location_of_functions = project_path+"code/utils/functions/" 
final_data            = "../data/"
behav_suffix           = "/behav/task001_run001/behavdata.txt"
t_data           =  final_data + 'glm/t_stat/'
p_data           =  final_data + 'glm/p-values/'


#sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
sys.path.append(location_of_functions)

sub_list = os.listdir(path_to_data)[1:]
# Progress bar
toolbar_width=len(sub_list)
sys.stdout.write("Conclusion:  ")
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    
from stimuli import events2neural
from event_related_fMRI_functions import hrf_single, convolution_specialized
from hypothesis import t_stat
from Image_Visualizing import present_3d, make_mask
from benjamini_hochberg import bh_procedure

#####################################
########## Clustering ##############
#####################################

#Mean across all subject

t_mean = np.zeros((64, 64, 34,24))

#loop through each person's T-statistic
count=0
for i in sub_list:

    t_stat = np.load(t_data+i+"_tstat.npy")
    mask = nib.load(path_to_data+i+'/anatomy/inplane001_brain_mask.nii.gz')
    mask_data = mask.get_data()

    t_mean[...,count] = make_mask(t_stat, mask_data, fit=True)
    count+=1

t_mean = np.mean(t_mean,axis=3)
final = present_3d(t_mean)
plt.imshow(final,interpolation='nearest', cmap='seismic')
plt.title("Mean T-Statistic Value Across 25 Subjects")

zero_out=max(abs(np.min(final)),np.max(final))
plt.clim(-10,10)
plt.colorbar()
plt.show()
plt.close()

#Cluster

data_new = t_mean[...,20:23]
X = np.reshape(data_new, (-1, 1))

connectivity = grid_to_graph(n_x= data_new.shape[0], n_y = data_new.shape[1], n_z = data_new.shape[2])

n_clusters = 8 # number of regions
ward = AgglomerativeClustering(n_clusters=n_clusters,
        linkage='ward', connectivity=connectivity).fit(X)
label = np.reshape(ward.labels_, data_new.shape)

label_mean = np.zeros(n_clusters)
center = list()

#FIND THE AVERAGE T-VALUE PER CLUSTER
for j in range(n_clusters):
    mask = label==j
    index = np.where(mask)
    center.append((np.mean(index[0]),np.mean(index[1]),np.mean(index[2])))
    label_mean[j] =np.mean(data_new[mask])

#PRINT THE PLOTS
for i in range(data_new.shape[-1]):
    plt.figure()
    plt.imshow(data_new[...,i], cmap=plt.cm.gray, interpolation ='nearest')
    for l in range(n_clusters):
        plt.contour(label[...,i] == l, contours=1,
            colors=[plt.cm.spectral(l / float(n_clusters)), ],linewidths= 0.4)
plt.xticks(())
plt.yticks(())
plt.show()



#####################################
####### MULTIPLE TESTING ############
#####################################
#
# p_mean = np.zeros((64, 64, 34,24))
#
# #loop through each person's T-statistic
# count=0
# for i in sub_list:
#
#     p_stat = np.load(p_data+i+"_pvalue.npy")
#     #mask = nib.load(path_to_data+i+'/anatomy/inplane001_brain_mask.nii.gz')
#     #mask_data = mask.get_data()
#
#     #p_mean[...,count] = make_mask(p_stat, mask_data, fit=True)
#
#     p_mean[...,count] = p_stat
#     count+=1
#
# p_mean = np.mean(p_mean,axis=3)/2
#
#
# p_vals = np.ravel(p_mean).T
#
# print("# ==== No Mask, bh_procedure ==== #")
# # a fairly large false discovery rate
# Q = .4
# significant_pvals = bh_procedure(p_vals, Q)
#
# reshaped_sig_p = np.reshape(significant_pvals, p_mean.shape)
#
# slice_reshaped_sig_p = reshaped_sig_p[...,7]
#
# plt.imshow(slice_reshaped_sig_p)
# plt.colorbar()
# plt.title('Significant p-values (No mask)')




