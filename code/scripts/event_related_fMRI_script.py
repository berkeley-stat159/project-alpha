# event_related_fMRI_event-related_fMRI_script.py
from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd # new
import sys # instead of os
import scipy.stats
from scipy.stats import gamma


#### location of desired files:
location_of_project="/Users/BenjaminLeRoy/Desktop/project-alpha/"
location_of_data=location_of_project+"data/ds009/" 
location_of_subject001=location_of_data+"sub001/" 
location_of_functions= location_of_project + "code/functions/"
location_of_our_data=location_of_project+"data/our_data/"
condition_location=location_of_subject001+"model/model001/onsets/task001_run001/"
bold_location=location_of_subject001+"BOLD/task001_run001/"

sys.path.append(location_of_functions) # 0
sys.path.append(bold_location) # 1
sys.path.append(condition_location) # 2



# 0. importing created convolution function for event-related fMRI functions:
from event_related_fMRI_functions import convolution
from event_related_fMRI_functions import hrf_single

# 1. load in subject001's BOLD data:
img=nib.load(location_of_subject001+"BOLD/task001_run001/"+"bold.nii")
data=img.get_data()
data=data[...,6:]
#data.shape

# 2. load in subject001's behavioral files (condition files)
cond1=np.loadtxt(condition_location+"cond001.txt")
cond2=np.loadtxt(condition_location+"cond002.txt")
cond3=np.loadtxt(condition_location+"cond003.txt")


# 3.
# some nasty looking shit to get a vector of stimuli changes, a 
# color vector to express which condition file they came from, and a
# vector of gaps between each stimuli instance

x=np.hstack((cond1[:,0],cond2[:,0],cond3[:,0]))
y=np.zeros((cond1.shape[0]+cond2.shape[0]+cond3.shape[0],))
y[cond1.shape[0]:]=1
y[(cond1.shape[0]+cond2.shape[0]):]+=1


xy=zip(x,y)
xy_sorted=sorted(xy,key= lambda x:x[0])

x_s,y_s=zip(*xy_sorted)

x_s_array=np.array([x for x in x_s])
desired=(x_s_array[1:]-x_s_array[:-1])

dictionary_color={0.:"red",1.:"blue",2.:"green"}
colors=[dictionary_color[elem] for elem in y_s[:-1]]




# 4.
# Looking at different potential convolution functions
# you should know why we can't use np.convolve :( by now
# all conditions together:
all_stimuli=np.array(sorted(list(cond2[:,0])+list(cond3[:,0])+list(cond1[:,0])))
all_stimuli_convolution = convolution(all_stimuli,np.ones(len(all_stimuli)),hrf_single)

plt.scatter(x_s_array[:-1],np.ones(len(x_s_array[:-1])),color=colors,label="stimuli instances")
plt.plot(all_stimuli,2*all_stimuli_convolution-1,label="HR response")
plt.xlim(0,475)
plt.xlabel("time")
plt.ylabel("Hemoglobin response")
plt.title("HR")

plt.scatter(x_s_array[:20],np.ones(20),color=colors,label="stimuli instances")
plt.plot(all_stimuli[:20],10*all_stimuli_convolution[:20],label="HR response")
plt.xlabel("time")
plt.ylabel("Hemoglobin response")
plt.title("HR zoomed")


# 5.
# Comparing our convolution function to a random voxel in our data

# standardizing values
test_time_standardized=(    test-np.mean(test)   )/scipy.std(test)    
asc_standardized=(   all_stimuli_convolution-np.mean(all_stimuli_convolution)   )/scipy.std(all_stimuli_convolution)

# plotting
plt.plot(2*np.arange(len(test)),test_time_standardized) 
plt.plot(all_stimuli, asc_standardized)
plt.xlabel("Time")
plt.title("Predicted Hemoglobin Response vs Actual")


# zooming
plt.plot(2*np.arange(len(test))[:50],test_time_standardized[:50]) 
index=np.zeros(len(all_stimuli))
index[all_stimuli<100]=1
plt.plot(all_stimuli[index==1.], asc_standardized[index==1.])
plt.xlabel("Time")
plt.title("Predicted Hemoglobin Response vs Actual, zoom")


