###############################################################################
# Presentation of new/final function, with enhanced speed and strong accuracy #
###############################################################################

#***********#
# Abstract: #
#***********#

	# We will present explainations of all 5 functions that attempt to approach 
	# event-related neural stimulus and the hemodynamic response in a clean way


######################
# Goals this script: #
######################

	# i. Can the user-created functions match np.convolve in np.convolve 
	#		territory
	# ii. Visialization of methods vs matching Neural Response
	# iii. Visialization of methods vs random voxel hemodynamic response. 
	

######################## 
# Name Standardization #
########################

	## Basic np.convolve:

	# 1) np : utilization of basic np.convolve function

	## My functions (mine and jane's now):

	# 2) second : uses the true stimulus values (doesn't standardize back to 
	#		image capture TR cuts) 
	# 3) third : improves off second, gives back information for correctly 
	#		spaced and desired image capture TR cuts 
	# 4) fourth : scraps 2 and 3, takes thinner time cuts then use np.convolve
	#		then rescales to the desired image capture TR cuts (speed gain)
	#		** with cuts =1 basically same approach as np.convolve **

	# 5) fifth : improves off third utilizes matrix multiplication (speed gain)



######################################
# Why did you go beyond np.convolve? #
######################################

	# Below we compare np.convolve for the event-related response to our 
	# functions

	###############
	# np.convolve #
	###############

	## PROS
	# 1. same length as data.shape[-1] (time dimension)
	# 2. fast, utilizes Fast Fourier Transform

	## CONS
	# 1. Doesn't take into account the variation of time instances
	# 2. Makes assumption of block stimulus



	####################
	# (my) convolution #
	####################

	## PROS
	# 1. Takes into account the strengths of event-based fMRI studies 
	# 		(variance allows for more views of the HRF in more detail)
	# 2. Doesn't make assumptions of the time a stimuli lasts, or 
	# 		length of time between events

	## CONS
	# 2. Slower (though we made significant gains from functions 2 and 3)


#########
# TL:DR #
#########

	# We developed a lot of functions to model the event-related stimulus, the
	# final models exibit high levels of similarities, The fasted model of these
	# models (when we restrict model 4 to 30 cuts) is model 5. With model 4 and
	# 15 cuts, model 4 wins the speed race. Either are optimized below 100 ms,
	# and as such, we are ok running 34 runs per person (with 24 people).

	# overall time cost: ~ .1*34*24/60 = 1.36 minutes for the whole run

#------------------------------------------------------------------------------#

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
from event_related_fMRI_functions import convolution, hrf_single 
from event_related_fMRI_functions import convolution_specialized
from event_related_fMRI_functions import np_convolve_30_cuts

from event_related_fMRI_functions import fast_convolution,fast_hrf

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
cond_all=np.loadtxt(condition_location+"cond_all.txt")





##############                ##############
############################################
# i. Comparing convolution and np.convolve #
############################################
##############                ##############

# i. Can the user-created functions match np.convolve in np.convolve territory

TR = 2.5
tr_times = np.arange(0, 30, TR)
hrf_at_trs = np.array([hrf_single(x) for x in tr_times])

n_vols = 173
neural_prediction = events2neural(location_to_class_data+'ds114_sub009_t2r1_cond.txt',TR,n_vols)
all_tr_times = np.arange(173) * TR


##################
# a. np.convolve #
##################


testconv_np = np.convolve(neural_prediction, hrf_at_trs) # hrf_at_trs sample data
N = len(neural_prediction)  # N == n_vols == 173
M = len(hrf_at_trs)  # M == 12
testconv_np=testconv_np[:N]

#####################
# b. user functions #
#####################

#--------#
# second #

testconv_2 = convolution(all_tr_times,neural_prediction,hrf_single)


#-------#
# third #

testconv_3 = convolution_specialized(all_tr_times,neural_prediction,
	hrf_single,all_tr_times)


#--------#
# fourth #

on_off = np.zeros(174)
real_times,on_off[:-1] = np.linspace(0,432.5,173+1),neural_prediction
hrf_function,TR,record_cuts= hrf_single, 2.5 ,np.linspace(0,432.5,173+1)
#
testconv_4_1 = np_convolve_30_cuts(real_times,on_off,hrf_function,TR,record_cuts,cuts=1)[0]

testconv_4_15 = np_convolve_30_cuts(real_times,on_off,hrf_function,TR,record_cuts,cuts=15)[0]


testconv_4_30 = np_convolve_30_cuts(real_times,on_off,hrf_function,TR,record_cuts,cuts=30)[0]


#-------#
# fifth #

testconv_5 = fast_convolution(all_tr_times,neural_prediction,fast_hrf,all_tr_times)



#######################
# c. Plot Comparision #
#######################

plt.plot(all_tr_times,testconv_np,label="conv_np")
plt.plot(all_tr_times,testconv_2,label="user 2")
plt.plot(all_tr_times,testconv_3,label="user 3")
plt.plot(np.linspace(0,432.5,174),testconv_4_1,label="user 4, cut = 1")
plt.plot(np.linspace(0,432.5,174),testconv_4_15,label="user 4, cut = 15")
plt.plot(np.linspace(0,432.5,174),testconv_4_30,label="user 4, cut = 30 (standard)")
plt.plot(all_tr_times,testconv_5,label="user 5")
plt.title("User-made functions matching np.convolve in np.convolve territory")
plt.xlabel("Time")
plt.ylabel("Predicted Hemoglobin response")
plt.legend(loc='lower right', shadow=True,fontsize="smaller")
plt.savefig(location_of_images+'test_comparision.png')
plt.close()

########################
# d. Timeit comparison #
########################

# Commentary on runs. You can see that conv_np gets really fast, this doesn't
# mirror the cost of running the function from the stimulus (gets around 3 ms) 


# runs from my (ben) computer:

	## testconv_np
	# In [1]: %timeit testconv_np = np.convolve(neural_prediction, hrf_at_trs) 
	# 	The slowest run took 4.67 times longer than the fastest. 
	# 	This could mean that an intermediate result is being cached 
	# 	100000 loops, best of 3: 12 µs per loop 

	## testconv_2	
	# In [2]: %timeit testconv_2 = convolution(all_tr_times,neural_prediction,hrf_single)
	# 	1 loops, best of 3: 812 ms per loop

	## testconv_3
	# In [3]: %timeit testconv_3 = convolution_specialized(all_tr_times,neural_prediction,hrf_single,all_tr_times)
	# 	1 loops, best of 3: 797 ms per loop

	## testconv_4_1
	# In [4]: %timeit testconv_4_1 = np_convolve_30_cuts(real_times,on_off,hrf_function,TR,record_cuts,cuts=1)[0]
	# 	100 loops, best of 3: 9.52 ms per loop

	## testconv_4_15
	# In [4]: %timeit testconv_4_15 = np_convolve_30_cuts(real_times,on_off,hrf_function,TR,record_cuts,cuts=15)[0]
	# 	10 loops, best of 3: 95.8 ms per loop

	## testconv_4_30
	# In [5]: %timeit testconv_4_30 = np_convolve_30_cuts(real_times,on_off,hrf_function,TR,record_cuts,cuts=30)[0]
	# 	10 loops, best of 3: 139 ms per loop

	## testconv_5
	# In [5]: %timeit testconv_5 = fast_convolution(all_tr_times,neural_prediction,fast_hrf)
	# 10 loops, best of 3: 102 ms per loop
test_names={"testconv_np": "12 µs (3 ms w/ stimulus)",
			"testconv_2": "812 ms",
			"testconv_3": "797 ms",
			"testconv_4_1": "9.52 ms",
			"testconv_4_15": "95.8 ms",
			"testconv_4_30": "139 ms",
			"testconv_5":"102 ms"}		

print("********")
print("Timings for %timeit:")
print(test_names)
print("********")

##############                              ##############
##########################################################
# ii. Comparision of the two functions (Neural Response) #
##########################################################
##############                              ##############


##################################################################################
# a. Creating the neural response (based on condition files - nonconstant gaps)  #
##################################################################################

# a kinda ugly way to get a sorted listing of stimulis time points and coloring
#  for the different conditions 
# (only 3 conditions), wouldn't generalize for other studies 
# (will we get to them?)  <- lolz old note

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
# b. np.convolve #
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
# c. user convolution #
#######################

# note: np.linspace(0,239*2-2,239) ==all_tr_times

cond_all=np.array(sorted(list(cond2[:,0])+list(cond3[:,0])+list(cond1[:,0]))) # could also just x_s_array

#--------#
# second #

conv_2 = convolution(cond_all,np.ones(len(cond_all)),hrf_single)
scaled_2=(conv_2-np.mean(conv_2))/(2*np.std(conv_2)) +.4


#-------#
# third #

conv_3 = convolution_specialized(cond_all,np.ones(len(cond_all)),hrf_single,np.linspace(0,239*2-2,239))
scaled_3=(conv_3-np.mean(conv_3))/(2*np.std(conv_3)) +.4


#--------#
# fourth #


real_times,on_off = cond_all,np.ones(len(cond_all))
hrf_function,TR,record_cuts= hrf_single, 2 ,np.linspace(0,239*2-2,239)

conv_4_15 = np_convolve_30_cuts(real_times,on_off,hrf_function,TR,record_cuts,cuts=15)[0]
scaled_4_15=(conv_4_15-np.mean(conv_4_15))/(2*np.std(conv_4_15)) +.4

conv_4_30 = np_convolve_30_cuts(real_times,on_off,hrf_function,TR,record_cuts,cuts=30)[0]
scaled_4_30=(conv_4_30-np.mean(conv_4_30))/(2*np.std(conv_4_30)) +.4

#-------#
# fifth #

conv_5 = fast_convolution(cond_all,np.ones(len(cond_all)),fast_hrf,np.linspace(0,239*2-2,239))
scaled_5=(conv_5-np.mean(conv_5))/(2*np.std(conv_5)) +.4




############################
# d. Plot for comparisions #
############################

plt.scatter(cond_all,np.zeros(len(cond_all)),color=colors,label="stimuli instances")
plt.plot(all_tr_times,scaled_np,label="np scaled")
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
# zoom: if you're exploring this yourself, don't run this (zoom in yourself silly :)  )
plt.xlim(0,50)
plt.ylim(-1,1.5)
plt.savefig(location_of_images+"convolution_vs_neural_stimulus.png")
plt.close()

########################
# d. Timeit comparison #
########################

# Commentary on runs. You can see that conv_np gets really fast, this doesn't
# mirror the cost of running the function from the stimulus (gets around 3 ms) 


# runs from my (ben) computer:

	## conv_np
	# In [1]: %timeit conv_np=np.convolve(neural_prediction,hrf_at_trs)
	# 	The slowest run took 6.01 times longer than the fastest. 
	#	This could mean that an intermediate result is being cached 
	# 	100000 loops, best of 3: 14.4 µs per loop

	## conv_2	
	# In [2]: %timeit conv_2 = convolution(cond_all,np.ones(len(cond_all)),hrf_single)
	# 	1 loops, best of 3: 972 ms per loop

	## conv_3
	# In [3]: %timeit conv_3 = convolution_specialized(cond_all,np.ones(len(cond_all)),hrf_single,np.linspace(0,239*2-2,239))
	# 	1 loops, best of 3: 1.15 s per loop

	## conv_4_15
	# In [4]: %timeit conv_4_15 = np_convolve_30_cuts(real_times,on_off,hrf_function,TR,record_cuts,cuts=15)[0]
	# 	10 loops, best of 3: 98.3 ms per loop

	## conv_4_30
	# In [5]: %timeit conv_4_30 = np_convolve_30_cuts(real_times,on_off,hrf_function,TR,record_cuts,cuts=30)[0]
	# 	10 loops, best of 3: 185 ms per loop

	## conv_5
	# In [6]: %timeit conv_5 = fast_convolution(cond_all,np.ones(len(cond_all)),fast_hrf,np.linspace(0,239*2-2,239))
	# 	10 loops, best of 3: 110 ms per loop
names={"conv_np": "14.4 µs (3 ms + w/ stimulus)",
			"conv_2": "972 ms",
			"conv_3": "1.15 s",
			"conv_4_15": "98.3 ms per loop",
			"conv_4_30": "185 ms",
			"conv_5": "110 ms"}		

print("********")
print("Timings for %timeit:")
print(names)
print("********")


##############                                     ##############
#################################################################
# iii. Comparision of the two functions (single voxel response) #
#################################################################
##############                                     ##############


############################################
# a. Pick a good voxel to compare against  #
############################################



from glm import glm
from Image_Visualizing import present_3d


beta_np,X_np=glm(data,conv_np)
# beta_2,X_2=glm(data,conv_2) not correct shape
beta_3,X_3=glm(data,conv_3)
beta_4,X_4=glm(data,conv_4_30)
#beta_5,X_5=glm(data,conv_5)


# non-np are stronger/more clear
plt.imshow(present_3d(beta_np[...,1]),cmap="gray",interpolation="nearest")
plt.imshow(present_3d(beta_3[...,1]),cmap="gray",interpolation="nearest")
plt.imshow(present_3d(beta_4[...,1]),cmap="gray",interpolation="nearest")
#plt.imshow(present_3d(beta_5[...,1]),cmap="gray",interpolation="nearest")


plt.imshow(beta_4[...,2,1],cmap="gray",interpolation="nearest")
plt.colorbar()
plt.close()

# From visual analysis
# In the regression has a really high beta_1 value at:
# beta_my[41,47,2,1] (voxel data[41,47,2] )
# lets use the comparisons (I know that is not good practice to check created 
#	X based on betas based on X)


###########################################
# b. Getting the voxel data standardized  #
###########################################

voxel_time_sequence = data[41,47,2]
voxel_time_standardized=(    voxel_time_sequence-np.mean(voxel_time_sequence)   )/scipy.std(voxel_time_sequence)    

#################################################################
# c. Plotting the convolution functions against voxel actual HR #
#################################################################


plt.plot(2*np.arange(len(voxel_time_sequence)),voxel_time_standardized,"-o",label="voxel actual HR") 
plt.scatter(cond_all,np.zeros(len(cond_all)),color=colors,label="stimuli instances")
plt.plot(all_tr_times,scaled_np,label="np scaled")
plt.plot(cond_all,scaled_2,label="user 2 scaled")
plt.plot(all_tr_times,scaled_3,"-o",label="user 3 scaled")
plt.plot(all_tr_times,scaled_4_30,"-*",label="user 4 scaled (30 cuts)",color="k")
plt.plot(all_tr_times,scaled_4_15,"-*",label="user 4 scaled (15 cuts)")
plt.plot(all_tr_times,scaled_5,"-.",label="user 5 scaled")
plt.xlabel("Time")
plt.title("Comparing predicted HR to real response for random voxel (standardized)")
plt.legend(loc='lower right', shadow=True,fontsize="smaller")
plt.xlim(0,50)
plt.ylim(-3,2)
plt.savefig(location_of_images+'convolution_vs_voxel_HR.png')
plt.close()

print("********")
print("Go check out those plots :) ")
print("********")







