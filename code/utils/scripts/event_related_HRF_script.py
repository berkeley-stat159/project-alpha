
########################################
########################################
# Initial Pros vs Cons for each method #
########################################
########################################

###############
# np.convolve #
###############

## PROS
# 1. same length as data.shape[-1] (time dimension)
# 2. fast utilizes Fast Fourier Transform

## CONS
# 1. Doesn't take into account the variation of time instances
# 2. Makes assumption of block stimulus

####################
# (my) convolution #
####################

## PROS
# 1. Takes into account the strengths of event-based fMRI studies (variance allows for more views of the HRF in more detail)
# 2. Doesn't make assumptions of the time a stimuli lasts, or length of time between events

## CONS
# 2. Slightly slower (not enough runs to really matter - 1 per subject per trial (24 * 6 max))


################
# Both methods #
################

# CONS
# 1. Both rely on provided hrf estimation
# 2. Both assume at independence of the hrf with respect to time and that it experiences linear addition

# 3. Currently, both assume all different types of conditions have the same hrf response (applitude and shape)


#*************************************************#
#*************************************************#

#*************************************************#
#*************************************************#


##############
# Begin code #
##############


# event_related_HRF_script.py
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


#######################################
# Creating locations of desired files #
#######################################

location_of_project="../../../"
location_of_data=location_of_project+"data/ds009/" 
location_of_subject001=location_of_data+"sub001/" 
location_of_functions= "../functions/"
location_of_our_data=location_of_project+"data/our_data/"
condition_location=location_of_subject001+"model/model001/onsets/task001_run001/"
bold_location=location_of_subject001+"BOLD/task001_run001/"
location_to_class_data=location_of_project+"data/ds114/"
location_of_images=location_of_project+"images/"


##########################
# Appending the sys path #
##########################

sys.path.append(location_of_functions) # 0
sys.path.append(bold_location) # 1
sys.path.append(condition_location) # 2
sys.path.append(location_to_class_data) # Goals: i


##########################
# Loading in basic files #
##########################

# 0.a importing created convolution function for event-related fMRI functions:
from event_related_fMRI_functions import convolution, hrf_single, convolution_specialized
# 0.b importing events2neural for np.convolve built-in function
from stimuli import events2neural

# 1. load in subject001's BOLD data:
img=nib.load(location_of_subject001+"BOLD/task001_run001/"+"bold.nii")
data=img.get_data()
data=data[...,6:]
#data.shape

# 2. load in subject001's behavioral files (condition files)
cond1=np.loadtxt(condition_location+"cond001.txt")
cond2=np.loadtxt(condition_location+"cond002.txt")
cond3=np.loadtxt(condition_location+"cond003.txt")


######################
######################
# Goals this script: #
######################
######################

# what would we like to see (as the debate between np.convolve and my convolution function):
# i. That the new function can do similar things (with similar assumptions that np.convolve can do)
# ii. Comparision of the two functions on actual data set with nueral response. (Neural Response)
# iii. Comparision of the two functions to a random voxel Hemoglobin response. (Actual Heomoglobin response)
 



##############                ##############
############################################
# i. Comparing convolution and np.convolve #
############################################
##############                ##############


TR = 2.5
tr_times = np.arange(0, 30, TR)
hrf_at_trs = np.array([hrf_single(x) for x in tr_times])

n_vols = 173
neural_prediction = events2neural(location_to_class_data+'ds114_sub009_t2r1_cond.txt',TR,n_vols)
all_tr_times = np.arange(173) * TR


##################
# a. np.convolve #
##################

convolved = np.convolve(neural_prediction, hrf_at_trs) # hrf_at_trs sample data
N = len(neural_prediction)  # N == n_vols == 173
M = len(hrf_at_trs)  # M == 12
convolved=convolved[:N]


################################
# b. convolution (my function) #
################################

my_convolved=convolution(np.linspace(0,432.5,173),neural_prediction,hrf_single)


#######################
# c. Plot Comparision #
#######################


plt.plot(np.linspace(0,432.5,173),convolved,label="np.convolved")
plt.plot(np.linspace(0,432.5,173),my_convolved,label="my convolution function")
plt.title("Examining if 'convolution' can do the same thing as 'np.convolve' ")
plt.xlabel("Time")
plt.ylabel("Predicted Hemoglobin response")
plt.legend(loc='lower right', shadow=True,fontsize="smaller")
plt.savefig(location_of_images+'test_comparision.png')
plt.close()
print("********")
print("i. c. Plot completed")
print("********")


max_diff= max(abs(convolved-my_convolved)) 
print("********")
print("Basic Test, max difference between 2 functions: " + str(max_diff))
print("********")



##############                              ##############
##########################################################
# ii. Comparision of the two functions (Neural Response) #
##########################################################
##############                              ##############


##################################################################################
# a. Creating the neural response (based on condition files - nonconstant gaps)  #
##################################################################################

# a kinda ugly way to get a sorted listing of stimulis time points and coloring for the different conditions 
# (only 3 conditions), wouldn't generalize for other studies (will we get to them?)
def create_stimuli_from_all_values(cond1,cond2,cond3):
	""" creates a sorted np.array for all stimulis in the condition files 
	
	Parameters:
	-----------
	three np.arrays (with the the times in the first column)

	Returns:
	--------
	x_s_array = a sorted np.array (1 dimensional) of all times in all condition files
	gap_between = the difference between t_i and t_{i+1}
	colors = list of color codes of the different times (corresponding to condition file number)

	"""


	x=np.hstack((cond1[:,0],cond2[:,0],cond3[:,0]))
	y=np.zeros((cond1.shape[0]+cond2.shape[0]+cond3.shape[0],))
	y[cond1.shape[0]:]=1
	y[(cond1.shape[0]+cond2.shape[0]):]+=1


	xy=zip(x,y)
	xy_sorted=sorted(xy,key= lambda x:x[0])

	x_s,y_s=zip(*xy_sorted)

	x_s_array=np.array([x for x in x_s])
	gap_between=(x_s_array[1:]-x_s_array[:-1])

	dictionary_color={0.:"red",1.:"blue",2.:"green"}
	colors=[dictionary_color[elem] for elem in y_s]

	return x_s_array, gap_between, colors
x_s_array, gap_between, colors =create_stimuli_from_all_values(cond1,cond2,cond3)




#######################
# b. (my) convolution #
#######################

all_stimuli=np.array(sorted(list(cond2[:,0])+list(cond3[:,0])+list(cond1[:,0]))) # could also just x_s_array
all_stimuli_convolution = convolution(all_stimuli,np.ones(len(all_stimuli)),hrf_single)
all_stimuli_convolution_best_length = convolution_specialized(all_stimuli,np.ones(len(all_stimuli)),hrf_single,np.linspace(0,239*2-2,239))



scaled_HR_convolution=(all_stimuli_convolution-np.mean(all_stimuli_convolution))/(2*np.std(all_stimuli_convolution)) +.4
scaled_HR_convolution_bl=(all_stimuli_convolution_best_length-np.mean(all_stimuli_convolution_best_length))/(2*np.std(all_stimuli_convolution_best_length)) +.4


##################
# c. np.convolve #
##################

# initial needed values
TR = 2
tr_times = np.arange(0, 30, TR)
hrf_at_trs = np.array([hrf_single(x) for x in tr_times])


# creating the .txt file for the events2neural function
cond_all=np.row_stack((cond1,cond2,cond3))
cond_all=sorted(cond_all,key= lambda x:x[0])

np.savetxt(condition_location+"cond_all.txt",cond_all)
neural_prediction=events2neural(condition_location+"cond_all.txt",2,239) # 1s are non special events


# doing the np.convolve
convolve_np=np.convolve(neural_prediction,hrf_at_trs)
convolve_np=convolve_np[:-(len(hrf_at_trs)-1)] #shorting convolution vector

all_tr_times = np.arange(data.shape[-1]) * TR

scaled_HR_convolve_np=(convolve_np-np.mean(convolve_np))/(2*np.std(convolve_np)) +.4



############################
# d. Plot for comparisions #
############################

plt.scatter(all_stimuli,np.zeros(len(all_stimuli)),color=colors,label="stimuli instances")
plt.plot(all_tr_times,scaled_HR_convolve_np,label="np.convolve scaled")
plt.plot(all_stimuli,scaled_HR_convolution,label="(my) convolution scaled")
plt.plot(all_tr_times,scaled_HR_convolution_bl,"-.",label="(my) convolution scaled equal spacing")
plt.xlim(0,475)
plt.xlabel("time")
plt.ylabel("Hemoglobin response")
plt.title("HR functions vs neural stimulus")
plt.legend(loc='lower right', shadow=True,fontsize="smaller")
# zoom: if you're exploring this yourself, don't run this (zoom in yourself silly :)  )
plt.xlim(0,50)
plt.ylim(-1,1.5)
plt.savefig(location_of_images+"convolution_vs_neural_stimulus.png")
plt.close()
print("********")
print("ii. d. Plot completed")
print("********")



##############                                     ##############
#################################################################
# iii. Comparision of the two functions (single voxel response) #
#################################################################
##############                                     ##############


############################################
# a. Pick a good voxel to compare against  #
############################################

# Remember the names of the of the two different methods
# my convolution: all_stimuli_convolution_best_length
# np.convolve:  convolve_np

from glm import glm
from Image_Visualizing import present_3d

beta_my,X_my=glm(data,all_stimuli_convolution_best_length)
beta_np,X_np=glm(data,convolve_np)

plt.imshow(present_3d(beta_my[...,1]),cmap="gray",interpolation="nearest")
plt.imshow(present_3d(beta_np[...,1]),cmap="gray",interpolation="nearest")


plt.imshow(beta_my[...,2,1],cmap="gray",interpolation="nearest")
plt.colorbar()
plt.close()

# From visual analysis
# In the regression has a really high beta_1 value at:
# beta_my[41,47,2,1] (voxel data[41,47,2] )
# lets use the comparisons (I know that is not good practice to check created X based on betas based on X)


###########################################
# b. Getting the voxel data standardized  #
###########################################

voxel_time_sequence = data[41,47,2]
voxel_time_standardized=(    voxel_time_sequence-np.mean(voxel_time_sequence)   )/scipy.std(voxel_time_sequence)    

#################################################################
# c. Plotting the convolution functions against voxel actual HR #
#################################################################


plt.plot(2*np.arange(len(voxel_time_sequence)),voxel_time_standardized,"-o",label="voxel actual HR") 
plt.plot(all_tr_times,scaled_HR_convolution_bl,"-.",label="(my) convolution scaled equal spacing")
plt.plot(all_stimuli,scaled_HR_convolution,"-",label="(my) convolution scaled")
plt.plot(all_tr_times,scaled_HR_convolve_np,label="np.convolve scaled")
plt.xlabel("Time")
plt.title("Comparing predicted HR to real response for random voxel (standardized)")
plt.legend(loc='lower right', shadow=True,fontsize="smaller")
plt.xlim(0,50)
plt.ylim(-3,2)
plt.savefig(location_of_images+'convolution_vs_voxel_HR.png')
plt.close()

print("********")
print("iii. c. Plot completed")
print("********")




