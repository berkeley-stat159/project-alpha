################
# New Analysis #
################

# Goal is to eliminate noise not related to the Hemodynamic response by using 
# 1) fourier (cosine/sine curves) to capture cyclic response and 
# 2) a basic drift function to capture linear time drift

#############
# Questions #
#############

# Questions brought about while working on the analysis:
# 1) should the glm for the fourier be somehow determined per individual 
# 	not per voxel? (is per voxel analysis overfitting the assumed random noise)
# 2) Maybe make fourier from averaging of all voxels per person? 
# 	Would this be strong enough and the right approach?


#------------------------------------------------------------------------------#


##################
# Load Libraries #
##################

from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd # new
import sys # instead of os
import scipy.stats
from scipy.stats import gamma
import os
import scipy.stats as stats

location_of_project="../../../"
location_of_data=location_of_project+"data/ds009/" 
location_of_subject001=location_of_data+"sub001/" 
location_of_functions= "../functions/"
location_of_our_data=location_of_project+"data/our_data/"
condition_location=location_of_subject001+"model/model001/onsets/task001_run001/"
bold_location=location_of_subject001+"BOLD/task001_run001/"
location_to_class_data=location_of_project+"data/ds114/"
location_of_images=location_of_project+"images/"


sys.path.append(location_of_functions) 
sys.path.append(bold_location) 
sys.path.append(condition_location) 
sys.path.append(location_to_class_data) # Basic loading



from event_related_fMRI_functions import convolution, hrf_single
from event_related_fMRI_functions import convolution_specialized
# ii. importing events2neural for np.convolve built-in function
from stimuli import events2neural
# iii. import glm_multiple for multiple regression
from glm import glm_multiple, glm_diagnostics
# iv. import image viewing tool
from Image_Visualizing import present_3d
from noise_correction import mean_underlying_noise, fourier_predict_underlying_noise,fourier_creation


#################
# Basic Loading #
#################

# load in subject001's BOLD data:
img=nib.load(location_of_subject001+"BOLD/task001_run001/"+"bold.nii")
data=img.get_data()
data=data[...,6:]
num_voxels=np.prod(data.shape[:-1])
n_vols = data.shape[-1]

TR = 2
all_tr_times = np.arange(n_vols) * TR


cond_all=np.loadtxt(condition_location+"cond_all.txt")

#------------------------------------------------------------------------------#

#################
# First Attempt #
#################

# First approach allowed for fourier strength to be fit to each voxel, 
#	potentially overcorrecting and masking some response to neural stimulation

# X matrix
X = np.ones((n_vols,6))
X[:,1]=convolution_specialized(cond_all[:,0],np.ones(len(cond_all)),hrf_single,all_tr_times)
X[:,2]=np.linspace(-1,1,num=X.shape[0]) #drift
X[:,3:]=fourier_creation(X.shape[0],3)[:,1:]

# modeling voxel hemodynamic response
beta,junk=glm_multiple(data,X)
MRSS, fitted, residuals = glm_diagnostics(beta, X, data)

# individual voxel analysis

plt.plot(all_tr_times,data[41, 47, 2],label="actual",color="b")
plt.plot(all_tr_times,fitted[41, 47, 2], label="predicted",color="r")
plt.title("Data for sub001, voxel [41, 47, 2],fourier 3 fit to voxel")
plt.xlabel("Time")
plt.ylabel("Hemodynamic response")
plt.legend(loc='upper right', shadow=True,fontsize="smaller")
plt.savefig(location_of_images+'noise_correction__fit_to_voxel_fitted.png')
plt.close()

plt.plot(all_tr_times,residuals[41, 47, 2],label="residuals",color="b")
plt.plot([0,max(all_tr_times)],[0,0],label="origin (residual=0)",color="k")
plt.title("Residual for sub001, voxel [41, 47, 2],fourier 3 fit to voxel")
plt.xlabel("Time")
plt.ylabel("Hemodynamic response residual")
plt.legend(loc='upper right', shadow=True,fontsize="smaller")
plt.savefig(location_of_images+'noise_correction_fit_to_voxel_residuals.png')
plt.close()



out=stats.probplot(residuals[41, 47, 2], dist="norm",plot=plt)
plt.title("Q-Q plot for sub001, voxel [41, 47, 2],fourier 3 fit to voxel")
plt.savefig(location_of_images+'noise_correction_fit_to_voxel_residuals_QQ.png')
plt.close()



##################
# Second Attempt #
##################

# Approaching the fourier modeling of cyclic on the individual level, by fitting
# the fourier beta values on the means of the single of all voxels


y_mean=mean_underlying_noise(data)
X_mean, MRSS_mean, fitted_mean,residuals_mean=fourier_predict_underlying_noise(y_mean,3)


# looking at that the y_mean and fitted_mean valuse
plt.plot(y_mean,label="voxel mean")
plt.plot(fitted_mean,label="fitted")
plt.title("Mean plot and Fourier Fitting")
plt.xlabel("Time")
plt.ylabel("Mean Hemodynamic response")
plt.legend(loc='lower right', shadow=True,fontsize="smaller")
plt.savefig(location_of_images+'noise_correction_mean_all_fitted.png')
plt.close()

#### using the fitted_mean

X_2 = np.ones((n_vols,4))
X_2[:,1]=convolution_specialized(cond_all[:,0],np.ones(len(cond_all)),hrf_single,all_tr_times)
drift= np.linspace(-1,1,num=X.shape[0])
X_2[:,2]=drift
X_2[:,3]=fitted_mean


# named after question (2)
beta_2,junk=glm_multiple(data,X_2)
MRSS_2, fitted_2, residuals_2 = glm_diagnostics(beta_2, X_2, data)

plt.plot(all_tr_times,data[41, 47, 2],label="actual",color="b")
plt.plot(all_tr_times,fitted_2[41, 47, 2], label="predicted",color="r")
plt.title("Data for sub001, voxel [41, 47, 2],fourier 3 fit to mean")
plt.xlabel("Time")
plt.ylabel("Hemodynamic response")
plt.legend(loc='upper right', shadow=True,fontsize="smaller")
plt.savefig(location_of_images+'noise_correction_mean_individual_fitted.png')
plt.close()



plt.plot(all_tr_times,residuals_2[41, 47, 2],label="residuals",color="b")
plt.plot([0,max(all_tr_times)],[0,0],label="origin (residual=0)",color="k")
plt.title("Residual for sub001, voxel [41, 47, 2],fourier 3 fit to mean")
plt.xlabel("Time")
plt.ylabel("Hemodynamic response residual")
plt.legend(loc='upper right', shadow=True,fontsize="smaller")
plt.savefig(location_of_images+'noise_correction_mean_individual_residuals.png')
plt.close()


out=stats.probplot(residuals_2[41, 47, 2], dist="norm",plot=plt)
plt.title("Q-Q plot for sub001, voxel [41, 47, 2],fourier 3 fit to mean")
plt.savefig(location_of_images+'noise_correction_mean_individual_residuals_QQ.png')
plt.close()


