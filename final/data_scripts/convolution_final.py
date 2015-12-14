"""
Script to create convolved hrf 


Does the correct method to convolve the condition files for the output as well as time shifting that is required.
All this is stored in matrices for the specific condition file and the shifts per time.

"""


import numpy as np
import os
import sys
import pandas as pd

# Relative path to subject 1 data

project_path          = "../../"
path_to_data          = project_path+"data/ds009/"
location_of_images    = project_path+"images/"
location_of_functions = project_path+"code/utils/functions/" 
final_data            = "../data/"
behav_suffix           = "/behav/task001_run001/behavdata.txt"


sys.path.append(location_of_functions)


sub_list = os.listdir(path_to_data)
sub_list = [i for i in sub_list if 'sub' in i]

# Progress bar
toolbar_width=len(sub_list)
sys.stdout.write("Convolution:  ")
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    
from event_related_fMRI_functions import hrf_single, np_convolve_30_cuts
from time_shift import make_shift_matrix, time_correct


#Create convolution HRF for each subjects
for i in sub_list:
        
    behav=pd.read_table(path_to_data+i+behav_suffix,sep=" ")
    num_TR = float(behav["NumTRs"])
    
    #load each condition file
    cond1=np.loadtxt(path_to_data+ i+ "/model/model001/onsets/task001_run001/cond001.txt")
    cond2=np.loadtxt(path_to_data+ i+ "/model/model001/onsets/task001_run001/cond002.txt")
    cond3=np.loadtxt(path_to_data+ i+ "/model/model001/onsets/task001_run001/cond003.txt")
    
    # Suppose that TR=2. We know this is not a good assumption.
    TR = 2
    tr_times = np.arange(0, 30, TR)
    hrf_at_trs = np.array([hrf_single(x) for x in tr_times])

    # creating the .txt file for the events2neural function
    cond_all=np.row_stack((cond1,cond2,cond3))
    cond_all=sorted(cond_all,key= lambda x:x[0])

    cond_all=np.array(cond_all)[:,0]
    
    delta_y=2*(np.arange(34))/34

    shifted_all=make_shift_matrix(cond_all,delta_y)
    shifted_1= make_shift_matrix(cond1[:,0],delta_y)
    shifted_2= make_shift_matrix(cond2[:,0],delta_y)
    shifted_3= make_shift_matrix(cond3[:,0],delta_y)
    
    
    #Required for the time correction
    def make_convolve_lambda(hrf_function,TR,num_TRs):
        convolve_lambda=lambda x: np_convolve_30_cuts(x,np.ones(x.shape[0]),hrf_function,TR,np.linspace(0,(num_TRs-1)*TR,num_TRs),15)
        
        return convolve_lambda
        
    convolve_lambda=make_convolve_lambda(hrf_single,TR,num_TR)
    
    #create HRF for all condition and each condition individually
    hrf_matrix_all=time_correct(convolve_lambda,shifted_all,num_TR)
    hrf_matrix_1=time_correct(convolve_lambda,shifted_1,num_TR)
    hrf_matrix_2=time_correct(convolve_lambda,shifted_2,num_TR)
    hrf_matrix_3=time_correct(convolve_lambda,shifted_3,num_TR)
    
    
    np.savetxt("../data/hrf/"+i+"_hrf_all.txt",hrf_matrix_all)
    np.savetxt("../data/hrf/"+i+"_hrf_1.txt",hrf_matrix_1)
    np.savetxt("../data/hrf/"+i+"_hrf_2.txt",hrf_matrix_2)
    np.savetxt("../data/hrf/"+i+"_hrf_3.txt",hrf_matrix_3)
    
    sys.stdout.write("-")
    sys.stdout.flush()
    
sys.stdout.write("\n")

# if we want to save all the hrf matrices:
# 1 loops, best of 3: 1min 4s per loop

    
    
    

