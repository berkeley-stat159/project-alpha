""" makes plots for the linear regression for the paper. 
Run with: 
    python lin_reg_plot.py
"""

# plots for paper:
# from glm_final.py

from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd # new
import sys # instead of os
import os
import numpy.linalg as npl
# Relative path to subject 1 data

project_path          = "../../"
path_to_data          = project_path+"data/ds009/"
location_of_images    = project_path+"images/"
location_of_functions = project_path+"code/utils/functions/" 
final_data            = project_path+"final/data/"
smooth_data           = final_data + 'smooth/'
hrf_data              = final_data + 'hrf/'

sys.path.append(location_of_functions)

sub_list = os.listdir(path_to_data)[1:]
sub_list = [i for i in sub_list if 'sub' in i]

from glm import glm_multiple, glm_diagnostics
from Image_Visualizing import present_3d, make_mask
from noise_correction import mean_underlying_noise, fourier_predict_underlying_noise,fourier_creation
from hypothesis import t_stat_mult_regression
from mask_phase_2_dimension_change import masking_reshape_start, masking_reshape_end


#Choose only first subject
i=sub_list[0]
j=15

img = nib.load(smooth_data+ i +"_bold_smoothed.nii")
data = img.get_data().astype(float)
    
n_vols = data.shape[-1]    
convolve = np.loadtxt(hrf_data+i+"_hrf_all.txt")

residual_final = np.zeros((data.shape))
t_final = np.zeros((data.shape[:-1]))

data_slice = data[:,:,j,:]


#Prepare mask for PCA
mask = nib.load(path_to_data+i+'/anatomy/inplane001_brain_mask.nii.gz')
mask_data = mask.get_data()
mask_data = make_mask(np.ones(data.shape[:-1]), mask_data, fit=True)
mask_data = mask_data!=0
mask_data = mask_data.astype(int)

###PCA SHIT###
#to_2d= masking_reshape_start(data,mask)
#X_pca= to_2d - np.mean(to_2d,0) - np.mean(to_2d,1)[:,None]
#cov = X_pca.T.dot(X_pca)
#U, S, V = npl.svd(cov)
#pca_addition= U[:,:6] # ~40% coverage


#Create design matrix
X = np.ones((n_vols,9))
X[:,1] = convolve[:,j]
X[:,2]=np.linspace(-1,1,num=X.shape[0]) #drift
X[:,3:] = fourier_creation(n_vols,3)[:,1:]

beta,t,df,p = t_stat_mult_regression(data_slice, X)


MRSS, fitted, residuals = glm_diagnostics(beta, X, data_slice)


###########################
# Fitted VS Residual Plot #
###########################

plt.scatter(fitted[30,40,:],residuals[30,40,:])
min_max=(np.min(fitted[30,40,:]),np.max(fitted[30,40,:]))

plt.plot([min_max[0],min_max[1]],[0,0],color="k")

plt.xlim(min_max[0],min_max[1])
plt.title("Subject 001, voxel: [30,40,15]")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.savefig(location_of_images+'Fitted_v_Residuals.png')
plt.close()


##############################
# Fitted vs Time Course Plot #
##############################

plt.plot(fitted[30,40,:],label="fitted")
plt.plot(data_slice[30,40],label="recorded fMRI")
plt.title("Subject 001, voxel: [30,40,15]")
plt.xlabel("Time Course")
plt.ylabel("Brain fMRI recording")
plt.legend(loc='lower right', shadow=True,fontsize="smaller")
plt.savefig(location_of_images+'Fitted_v_Actual.png')
plt.close()



convolve_all = np.loadtxt(hrf_data+i+"_hrf_all.txt")[:,0]
convolve_1 = np.loadtxt(hrf_data+i+"_hrf_1.txt")[:,0]
convolve_2 = np.loadtxt(hrf_data+i+"_hrf_2.txt")[:,0]
convolve_3 = np.loadtxt(hrf_data+i+"_hrf_3.txt")[:,0]


#######################################
# Convolution based on each condition #
#######################################

xx=np.arange(len(convolve_all))*2
plt.plot(xx,convolve_all,color="#4000ff",label="All Conditions")
plt.plot(xx[[0,len(xx)-1]],[0,0],color="k")
plt.plot(xx,convolve_1-2,color="#0070ff",label="Condition 1")
plt.plot(xx[[0,len(xx)-1]],[-2,-2],color="k")
plt.plot(xx,convolve_2-4,color="#00bfff",label="Condition 2")
plt.plot(xx[[0,len(xx)-1]],[-4,-4],color="k")
plt.plot(xx,convolve_3-6,color="#00ffff",label="Condition 3")
plt.plot(xx[[0,len(xx)-1]],[-6,-6],color="k")
plt.title("Subject 001, first slice")
plt.xlabel("Time Course")
plt.yticks([])
plt.ylabel("Convolution")
plt.legend(loc='center right', shadow=True,fontsize="smaller")
plt.savefig(location_of_images+'all_cond_time.png')
plt.close()

