"""
Script to create "convolution_vs_neural_stimulus.png" 

"""
# event_related_HRF_script.py
from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import sys # instead of os
import os


#######################################
# Creating locations of desired files #
#######################################

location_of_project="../../"
location_of_data=location_of_project+"data/ds009/" 
location_of_subject001=location_of_data+"sub001/" 
location_of_functions = location_of_project+"code/utils/functions/" 
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
from event_related_fMRI_functions import convolution, hrf_single, convolution_specialized, np_convolve_30_cuts, fast_convolution,fast_hrf

# 0.b importing events2neural for np.convolve built-in function
from stimuli import events2neural


# 1. load in subject001's BOLD data:
img=nib.load(location_of_subject001+"BOLD/task001_run001/"+"bold.nii.gz")
data=img.get_data()
data=data[...,6:]
#data.shape


# 2. load in subject001's behavioral files (condition files)
cond1=np.loadtxt(condition_location+"cond001.txt")
cond2=np.loadtxt(condition_location+"cond002.txt")
cond3=np.loadtxt(condition_location+"cond003.txt")
cond_all=np.loadtxt(condition_location+"cond_all.txt")



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



##################
#    np.convolve #
##################

# initial needed values
TR = 2
tr_times = np.arange(0, 30, TR)
hrf_at_trs = np.array([hrf_single(x) for x in tr_times])


# creating the .txt file for the events2neural function
cond_all=np.row_stack((cond1,cond2,cond3))
cond_all=sorted(cond_all,key= lambda x:x[0])


np.savetxt(condition_location+"cond_all.txt",cond_all)
neural_prediction=events2neural(condition_location+"cond_all.txt",2,239) 
# 1s are non special events


# doing the np.convolve
conv_np=np.convolve(neural_prediction,hrf_at_trs)
conv_np=conv_np[:-(len(hrf_at_trs)-1)] #shorting convolution vector

all_tr_times = np.arange(data.shape[-1]) * TR

scaled_np=(conv_np-np.mean(conv_np))/(2*np.std(conv_np)) +.4



#######################
#    user convolution #
#######################

# note: np.linspace(0,239*2-2,239) ==all_tr_times

cond_all=np.array(sorted(list(cond2[:,0])+list(cond3[:,0])+list(cond1[:,0]))) # could also just x_s_array

#--------#
# second #
#--------#

conv_2 = convolution(cond_all,np.ones(len(cond_all)),hrf_single)
scaled_2=(conv_2-np.mean(conv_2))/(2*np.std(conv_2)) +.4


#-------#
# third #
#-------#

conv_3 = convolution_specialized(cond_all,np.ones(len(cond_all)),hrf_single,np.linspace(0,239*2-2,239))
scaled_3=(conv_3-np.mean(conv_3))/(2*np.std(conv_3)) +.4


#--------#
# fourth #
#--------#


real_times,on_off = cond_all,np.ones(len(cond_all))
hrf_function,TR,record_cuts= hrf_single, 2 ,np.linspace(0,239*2-2,239)

conv_4_15 = np_convolve_30_cuts(real_times,on_off,hrf_function,TR,record_cuts,cuts=15)
scaled_4_15=(conv_4_15-np.mean(conv_4_15))/(2*np.std(conv_4_15)) +.4

conv_4_30 = np_convolve_30_cuts(real_times,on_off,hrf_function,TR,record_cuts,cuts=30)
scaled_4_30=(conv_4_30-np.mean(conv_4_30))/(2*np.std(conv_4_30)) +.4

#-------#
# fifth #
#-------#

conv_5 = fast_convolution(cond_all,np.ones(len(cond_all)),fast_hrf,np.linspace(0,239*2-2,239))
scaled_5=(conv_5-np.mean(conv_5))/(2*np.std(conv_5)) +.4




############################
#    Plot for comparisions #
############################

plt.scatter(cond_all,np.zeros(len(cond_all)),color=colors,label="stimuli instances")
plt.plot(all_tr_times,scaled_np,label="np naive approach scaled")
plt.plot(cond_all,scaled_2,label="user 2 scaled")
plt.plot(all_tr_times,scaled_3,"-o",label="user 3 scaled")
plt.plot(all_tr_times,scaled_4_30,"-*",label="user 4 scaled (30 cuts)",color="k")
plt.plot(all_tr_times,scaled_4_15,"-*",label="user 4 scaled (15 cuts)")
plt.plot(all_tr_times,scaled_5,"-.",label="user 5 scaled")
plt.xlim(0,475)
plt.xlabel("time")
plt.ylabel("Hemoglobin response")
plt.title("HR functions vs neural stimulus")
plt.legend(loc='lower right', shadow=True,fontsize="smaller")
plt.xlim(0,50)
plt.ylim(-1,1.5)
plt.savefig(location_of_images+"convolution_vs_neural_stimulus.png")
plt.close()
