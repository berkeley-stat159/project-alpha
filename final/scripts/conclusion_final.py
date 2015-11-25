

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
plt.clim(-zero_out,zero_out)
plt.colorbar()
plt.show()

#Cluster

data_new = t_mean[...,10:15]
X = np.reshape(data_new, (-1, 1))

connectivity = grid_to_graph(n_x= data_new.shape[0], n_y = data_new.shape[1], n_z = data_new.shape[2])

n_clusters = 7 # number of regions
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
    plt.imshow(data_new[...,i], cmap=plt.cm.gray)
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
#
# for i in sub_list:
#     df=275
#     t = np.load(t_data+i+"_tstat.npy")
#     ltp = t_dist.cdf(abs(t), df)
#     p = 1-ltp
#
#     significant_pvalues = bh_procedure(p, .25)
#
#     significant_pvalues = significant_pvalues.reshape(t_mean.shape)
#
#     final = present_3d(significant_pvalues)
#
#     plt.imshow(final,interpolation='nearest', cmap='seismic')
#     plt.title("Significant P-values")
#
#     zero_out=max(abs(np.min(final)),np.max(final))
#     plt.clim(-zero_out,zero_out)
#     plt.colorbar()
