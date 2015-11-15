# multi_regression_script.py

# In this file we will be creating multiple regressions using the the glm 
# function. Moreover, the features added with be: seperating the conditions
# from each other (i.e. x_1 = cond1 HRF, x_2 = cond2 HRF, and x_3 = cond3 HRF.

# I will be running it with np.convolve and convolution_specialized.



# Steps:
# 1. Libraries, Location and Data
# 2. X matrix creation,  specifically creation of column vectors for X matrix 
# (a) np.convolve and (b) convolution_specialized
# 3. use glm to generate a linear regression


###################################
# 1. Libraries, Location and Data #
###################################

################
# a. Libraries #
################

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

################
################
# b. Locations #
################
################

location_of_project="../../../"
location_of_data=location_of_project+"data/ds009/" 
location_of_subject001=location_of_data+"sub001/" 
location_of_functions= "../functions/"
location_of_our_data=location_of_project+"data/our_data/"
condition_location=location_of_subject001+"model/model001/onsets/task001_run001/"
bold_location=location_of_subject001+"BOLD/task001_run001/"
location_to_class_data=location_of_project+"data/ds114/"
location_of_images=location_of_project+"images/"


###############
# c. sys path #
###############

sys.path.append(location_of_functions) # 0
sys.path.append(bold_location) # 1
sys.path.append(condition_location) # 2
sys.path.append(location_to_class_data) # Goals: i


################
# d. functions #
################

# i. importing created convolution function for event-related fMRI functions:
from event_related_fMRI_functions import convolution, hrf_single
from event_related_fMRI_functions import convolution_specialized
# ii. importing events2neural for np.convolve built-in function
from stimuli import events2neural
# iii. import glm_multiple for multiple regression
from glm import glm_multiple, glm_diagnostics
# iv. import image viewing tool
from Image_Visualizing import present_3d


###########
# e. data #
###########

# i. load in subject001's BOLD data:
img=nib.load(location_of_subject001+"BOLD/task001_run001/"+"bold.nii")
data=img.get_data()
data=data[...,6:]
num_voxels=np.prod(data.shape[:-1])
#data.shape

# ii. load in subject001's behavioral files (condition files) 
# for convolution_specialized
cond1=np.loadtxt(condition_location+"cond001.txt")
cond2=np.loadtxt(condition_location+"cond002.txt")
cond3=np.loadtxt(condition_location+"cond003.txt")

########################
########################
# 2. X Matrix Creation #  
########################
########################

###################
# (a) np.convolve #
###################

TR = 2
tr_times = np.arange(0, 30, TR)
hrf_at_trs = np.array([hrf_single(x) for x in tr_times])

n_vols = data.shape[-1] # time slices
X_np = np.ones((n_vols,4))

cond_string = ["cond001.txt","cond002.txt","cond003.txt"]
for i,name in enumerate(cond_string):
	nueral_prediction = events2neural(condition_location+name,TR,n_vols)
	hrf_long = np.convolve(nueral_prediction, hrf_at_trs)
	X_np[:,i+1] = hrf_long[:-(len(hrf_at_trs)-1)]

all_tr_times = np.arange(n_vols) * TR


###############################
# (b) convolution_specialized #
###############################
X_my = np.ones((n_vols,4))

conds = [cond1[:,0],cond2[:,0],cond3[:,0]]
for i,cond in enumerate(conds):
	X_my[:,i+1]=convolution_specialized(cond,np.ones(len(cond)),hrf_single,all_tr_times)

##########
##########
# 3. GLM #  
##########
##########

###################
# (a) np.convolve #
###################

B_np,junk=glm_multiple(data,X_np)

###############################
# (b) convolution_specialized #
###############################


B_my,junk=glm_multiple(data,X_my)



#############
# 4. Review #
#############

# Looks like splitting up the conditions does a few things
# 1. cond2 (exploding the balloon) have 2 effects, it continues the views from 
# 	all conditions to some extent and also (because it occurs so rarely- see 
#	first plot), that also takes in the shifting of the brain on the outside
# 2. cond3's beta is centered around 20 (so it may not be able to pick enought 
#	up)

plt.plot(X_np[:,2])
plt.title("Condition 2 (pop) time predictions")
plt.xlabel("Time")
plt.ylabel("Hemoglobin response")
plt.savefig(location_of_images+'cond2_time.png')
plt.close()

plt.imshow(present_3d(B_np[...,2]),interpolation='nearest', cmap='seismic') 
# instead of cmap="gray"
plt.title("Condition 2 (pop) beta Brain Image")
plt.colorbar()
zero_out=max(abs(np.min(present_3d(B_np[...,2]))),np.max(present_3d(B_np[...,2])))
plt.clim(-zero_out,zero_out)
plt.savefig(location_of_images+'mr_cond2_beta_brain.png')
plt.close()


plt.plot(X_np[:,3])
plt.title("Condition 3 (save) time predictions")
plt.xlabel("Time")
plt.ylabel("Hemoglobin response")
plt.savefig(location_of_images+'mr_cond3_time.png')
plt.close()

plt.imshow(present_3d(B_np[...,3]),interpolation='nearest', cmap='seismic')
# instead of cmap="gray"
zero_out=max(abs(np.min(present_3d(B_np[...,3]))),np.max(present_3d(B_np[...,3])))
plt.clim(-zero_out,zero_out)
plt.title("Condition 3 (save) beta Brain Image")
plt.colorbar()
plt.savefig(location_of_images+'mr_cond3_beta_brain.png')
plt.close()

plt.plot(X_np[:,1])
plt.title("Condition 1 time predictions")
plt.xlabel("Time")
plt.ylabel("Hemoglobin response")
plt.savefig(location_of_images+'mr_cond1_time.png')
plt.close()

plt.imshow(present_3d(B_np[...,1]),interpolation='nearest', cmap='seismic')
# instead of cmap="gray"
plt.title("Condition 1 beta Brain Image")
plt.colorbar()
plt.savefig(location_of_images+'mr_cond1_beta_brain.png')
plt.close()


difference_12=present_3d(B_np[...,1])-present_3d(B_np[...,2])
plt.imshow(difference_12,interpolation='nearest', cmap='seismic')
plt.title("Differences between Condition 1 and 2")
zero_out=max(abs(np.min(difference_12)),np.max(difference_12))
plt.clim(-zero_out,zero_out)
plt.colorbar()
plt.savefig(location_of_images+'mr_cond1-cond2_beta_brain.png')
plt.close()


plt.plot(X_np[:,1]+X_np[:,2]+X_np[:,3],label="All Conditions",color="#000019")
plt.plot([0,239],[0,0])
colors=["#000099","#1A1AFF","#9999FF"]
for i in range(3):
	plt.plot(X_np[:,(i+1)]-2*(i+1),label="Condition " +str(i+1),color=colors[i])
	plt.plot([0,239],[-2*(i+1),-2*(i+1)],color="#FF0000")

plt.legend(loc='center right', shadow=True,fontsize="smaller")

plt.title("Hemogoblin predicted response for different conditions")
plt.xlabel("Time")
plt.ylabel("Hemoglobin response")
plt.savefig(location_of_images+'all_cond_time.png')
plt.close()







MRSS_my, fitted_my, residuals_my = glm_diagnostics(B_my, X_my, data)
print("MRSS using multiple regression: "+str(np.mean(MRSS_my)))
plt.plot(data[41, 47, 2],label="actual HR response")
plt.plot(fitted_my[41, 47, 2],label="predicted HR response")
plt.title("Subject 001, voxel (41,47,2) HR Fitted vs actual")
plt.legend(loc='upper left', shadow=True,fontsize="smaller")
plt.savefig(location_of_images+"fitted_vs_actual_mult_regression.png")
plt.close()

