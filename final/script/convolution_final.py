# load condition
# cond_all

from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import sys 
import pandas as pd

# Progress bar
toolbar_width=len(sub_list)
sys.stdout.write("Convolution, with 'fwhm = 1.5':  ")
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

pathtodata = "../../../data/ds009/"
sub_list = os.listdir(pathtodata)[1:]
behav_suffix="/behav/task001_run001/behavdata.txt"

    
    
for i in os.listdir(pathtodata)[1:]:
    
	behav=pd.read_table(pathtodata+i+behav_suffix,sep=" ")
	num_TR = float(behav["NumTRs"])
    
    condition_location  = = nib.load(pathtodata+ i+ "/BOLD/task001_run001/bold.nii.gz")
    data = img.get_data()
    data = data[...,6:] 
    
    # Suppose that TR=2. We know this is not a good assumption.
    # Also need to look into the hrf function. 
    cond1=np.loadtxt(pathtodata+ i+ "/model/model001/onsets/task001_run001/cond001.txt")
    cond2=np.loadtxt(pathtodata+ i+ "/model/model001/onsets/task001_run001/cond002.txt")
    cond3=np.loadtxt(pathtodata+ i+ "/model/model001/onsets/task001_run001/cond003.txt")
    
    TR = 2
    tr_times = np.arange(0, 30, TR)
    hrf_at_trs = np.array([hrf_single(x) for x in tr_times])
    n_vols=data.shape[-1]

    # creating the .txt file for the events2neural function
    cond_all=np.row_stack((cond1,cond2,cond3))
    cond_all=sorted(cond_all,key= lambda x:x[0])

    cond_all=np.array(cond_all)[:,0]
    
    delta_y=2*(np.arange(34))/34


    shifted=make_shift_matrix(cond_all,delta_y)
    
    def make_convolve_lambda(hrf_function,TR,num_TRs):
        convolve_lambda=lambda x: np_convolve_30_cuts(x,np.ones(x.shape[0]),hrf_function,TR,np.linspace(0,(num_TRs-1)*TR,num_TRs),15)[0]
        return convolve_lambda
        
    convolve_lambda=make_convolve_lambda(hrf_single,TR,num_TR)
    
    hrf_matrix=time_correct(convolve_lambda,shifted,num_TR)
    
    np.savetxt("hrf.txt",hrf_matrix)
    
    
    
    

