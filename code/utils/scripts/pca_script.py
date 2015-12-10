"""
Script to do SVD on the covariance matrix of the voxel by time matrix.

Run with: 
    python pca_script.py

"""

import numpy as np
import numpy.linalg as npl
import nibabel as nib
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Relative paths to project and data. 
project_path          = "../../../"
path_to_data          = project_path+"data/ds009/"
location_of_images    = project_path+"images/"
location_of_functions = project_path+"code/utils/functions/" 
behav_suffix           = "/behav/task001_run001/behavdata.txt"

sys.path.append(location_of_functions)
from Image_Visualizing import make_mask

# List of subject directories.
sub_list = os.listdir(path_to_data)
sub_list = [i for i in sub_list if 'sub' in i]

# Initialize array to store variance proportions. 
masked_var_array = np.zeros((50, len(sub_list)))

# Loop through all the subjects. 
for j in range(len(sub_list)):
    name = sub_list[j]
    # amount of beginning TRs not standardized at 6
    behav=pd.read_table(path_to_data+name+behav_suffix,sep=" ")
    num_TR = float(behav["NumTRs"])
    
    # Load image data.
    img = nib.load(path_to_data+ name+ "/BOLD/task001_run001/bold.nii.gz")
    data = img.get_data()
    data = data.astype(float) 

    # Load mask.
    mask = nib.load(path_to_data+ name+'/anatomy/inplane001_brain_mask.nii.gz')
    mask_data = mask.get_data()

    # Drop the appropriate number of volumes from the beginning. 
    first_n_vols=data.shape[-1]
    num_TR_cut=int(first_n_vols-num_TR)
    data = data[...,num_TR_cut:] 

    # Now fit a mask to the 3-d image for each time point.
    my_mask = np.zeros(data.shape)
    for i in range(my_mask.shape[-1]): 
         my_mask[...,i] = make_mask(data[...,i], mask_data, fit=True)
    
    # Reshape stuff to 2-d (voxel by time) and mask the data. 
    # This should cut down the number of volumes by more than 50%.
    my_mask_2d = my_mask.reshape((-1,my_mask.shape[-1]))
    data_2d = data.reshape((-1,data.shape[-1]))
    masked_data_2d = data_2d[my_mask_2d.sum(1) != 0,:]

    # Subtract means over voxels (columns).
    data_2d = data_2d - np.mean(data_2d, 0)
    masked_data_2d = masked_data_2d - np.mean(masked_data_2d, 0)

    # Subtract means over time (rows)
    data_2d = data_2d - np.mean(data_2d, axis=1)[:, None]
    masked_data_2d = masked_data_2d - np.mean(masked_data_2d, axis=1)[:, None]

    # PCA analysis on unmasked data: 
    # Do SVD on the time by time matrix and get explained variance.
    U, S, VT = npl.svd(data_2d.T.dot(data_2d))
    exp_var = S / np.sum(S)
    var_sums = np.cumsum(exp_var)

    # PCA analysis on MASKED data: 
    # Do SVD on the time by time matrix and get explained variance.
    U_masked, S_masked, VT_masked = npl.svd(masked_data_2d.T.dot(masked_data_2d))
    exp_var_masked = S_masked / np.sum(S_masked)
    var_sums_masked= np.cumsum(exp_var_masked)
    masked_var_array[:,j] = exp_var_masked[:50] # Store the first 50 variance proportions.
    
    # Setting up legend colors.
    hand_un = mlines.Line2D([], [], color='b', label='Not Masked')
    hand_mask = mlines.Line2D([], [], color='r', label='Masked')

    # Compare proportion of variance explained by each component for masked and unmasked data.
    plt.plot(exp_var[np.arange(1,11)], 'b-o')
    plt.plot(exp_var_masked[np.arange(1,11)], 'r-o')
    plt.legend(handles=[hand_un, hand_mask])
    plt.xlabel("Principal Components")
    plt.title("Proportion of Variance Explained by Each Component for " + name)
    plt.savefig(location_of_images+'pcapropvar'+name+'.png')
    plt.close()

    # Compare sum of proportion of variance explained by each component for masked and unmasked data.
    plt.plot(var_sums[np.arange(1,11)], 'b-o')
    plt.plot(var_sums_masked[np.arange(1,11)], 'r-o')
    plt.axhline(y=0.4, color='k')
    plt.legend(handles=[hand_un, hand_mask])
    plt.xlabel("Number of Principal Components")
    plt.title("Sum of Proportions of Variance Explained by Components for " + name)
    plt.savefig(location_of_images+'pcacumsums'+name+'.png')
    plt.close()

# Write array of variance proportions to a text file.
masked_var = pd.DataFrame(masked_var_array)
cumsums = masked_var.cumsum(0)


#######################
# Plots of Components #
#######################
plt.plot(np.arange(1,11), cumsums.median(1)[:10], 'r-o')
plt.grid()
plt.axhline(y=0.4, color='k', linestyle="--")
plt.xlabel("Principal Components")
plt.title("Sum of Proportions of Variance Explained by Components")
plt.savefig(location_of_images+'pcaALL.png')
plt.close()


##########################
# Boxplots of components #
##########################
plt.boxplot(np.array(cumsums[:10]).T)
plt.scatter(np.ones((24,10))*np.arange(1,11), np.array(cumsums[:10]).T)
plt.grid()
plt.axhline(y=0.4, color='k', linestyle="--")
plt.xlabel("Principal Components")
plt.title("Sum of Proportions of Variance Explained by Components")
plt.savefig(location_of_images+'pcaBOX.png')


