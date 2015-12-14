import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "functions"))

import time as time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from Image_Visualizing import present_3d, make_mask


# Relative path to subject 1 data
pathtodata = "../../../data/ds009/sub001/"
condition_location=pathtodata+"model/model001/onsets/task001_run001/"
location_of_images="../../../images/"

sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))


data = np.load('cluster_mask.npy')

data_new = data[..., 10:13]

X = np.reshape(data_new, (-1, 1))

connectivity = grid_to_graph(n_x= data_new.shape[0], n_y = data_new.shape[1], n_z = data_new.shape[2])

st = time.time()
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
    plt.savefig(location_of_images+"ward"+str(i)+'.png')
    




