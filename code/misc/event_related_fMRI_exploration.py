# event_related_fMRI_event-related_fMRI_exploration.py
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


# plotting (not important enough to save, sorry)
plt.scatter(x_s_array[:-1],desired,color=colors)
plt.xlabel("stimuli")
plt.ylabel("time before next stimulation")
plt.title("Stimuli Instances")

plt.scatter(x_s_array[:20],desired[:20],color=colors)
plt.plot(cond1[:20,0],10*x[:20])
plt.xlabel("stimuli")
plt.ylabel("time before next stimulation")
plt.title("Stimuli, Zoomed in")


# 4.
# Looking at different potential convolution functions
# you should know why we can't use np.convolve :( by now

## (a) cond1
basic_c1 = cond1[:,0]
basic_c1_convolution = convolution(basic_c1,np.ones(len(basic_c1)),hrf_single)

## plotting

plt.scatter(x_s_array[:-1],desired,color=colors,label="stimuli instances")
plt.plot(basic_c1,10*basic_c1_convolution,label="HR response")
plt.xlim(0,475)
plt.xlabel("time")
plt.ylabel("time before next stimulation/ Hemoglobin response")
plt.title("HR, response for cond1")


plt.scatter(x_s_array[:20],desired[:20],color=colors,label="stimuli instances")
plt.plot(basic_c1[:20],10*basic_c1_convolution[:20],label="HR response")
plt.xlabel("time")
plt.ylabel("time before next stimulation/ Hemoglobin response")
plt.title("HR, response for cond1, zoomed in")




## (b) cond2 and cond3
non_basic_b = np.array(sorted(list(cond2[:,0])+list(cond3[:,0])))
non_basic_b_convolution = convolution(non_basic_b,np.ones(len(non_basic_b)),hrf_single)


## plotting
plt.scatter(x_s_array[:-1],np.ones(len(x_s_array[:-1])),color=colors,label="stimuli instances")
plt.plot(non_basic_b,2*non_basic_b_convolution+1.2,label="HR response")
plt.xlim(0,475)
plt.xlabel("time")
plt.ylabel("Hemoglobin response")
plt.title("HR, response for cond2 and cond3 only")


## (c) all conditions
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



# 6.
# Comparing our convolution function to general convolution run (using only the first condition)
# this doesn't really even make sense now, but what the heck:
from stimuli import events2neural

neural_prediction=events2neural(condition_location+"cond001.txt",2,239) # 1s are non special events

def hrf(times):
    """ Return values for HRF at given times """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6   # this max can mess up size of model (which do we have it)


TR = 2
tr_times = np.arange(0, 30, TR)
hrf_at_trs = hrf(tr_times)


convolved=np.convolve(neural_prediction,hrf_at_trs)
convolved=convolved[:-(len(hrf_at_trs)-1)]

all_tr_times = np.arange(data.shape[-1]) * TR

plt.plot(all_tr_times, neural_prediction)
plt.plot(all_tr_times, convolved)

scaled_HR_basic_c1=(basic_c1_convolution-np.mean(basic_c1_convolution))/(2*np.std(basic_c1_convolution)) +.5

# too much stuff!!!!!!
plt.scatter(x_s_array[:-1],np.ones(len(x_s_array[:-1])),color=colors,label="stimuli instances")
plt.plot(basic_c1,scaled_HR_basic_c1,label="HR response scaled")
plt.plot(all_tr_times, neural_prediction)
plt.plot(all_tr_times, convolved)


# zoomed, awwwwww
plt.scatter(x_s_array[:20],np.ones(len(x_s_array[:20])),color=colors,label="stimuli instances")
plt.plot(basic_c1[:20],scaled_HR_basic_c1[:20],label="HR response scaled")
plt.plot(all_tr_times[:20], neural_prediction[:20], label="step function")
plt.plot(all_tr_times[:20], convolved[:20],label="basic convolution HR")

