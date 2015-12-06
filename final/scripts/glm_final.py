"""
Script to run glm

Design matrix takes into account conditions, drift, fourier and pca

Also runs slice by slice in order to correct for time

For each subject, this function writes the data files for the t-values, p-values, and residuals
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd # new
import sys # instead of os
import os


# Relative path to subject all of the subjects

project_path          = "../../"
path_to_data          = project_path+"data/ds009/"
location_of_images    = project_path+"images/"
location_of_functions = project_path+"code/utils/functions/" 
final_data            = "../data/"
behav_suffix           = "/behav/task001_run001/behavdata.txt"
smooth_data           =  final_data + 'smooth/'
hrf_data              = final_data + 'hrf/'


sys.path.append(location_of_functions)

sub_list = os.listdir(path_to_data)[1:]

#Import our functions
from glm import glm_multiple, glm_diagnostics
from Image_Visualizing import present_3d
from noise_correction import mean_underlying_noise, fourier_predict_underlying_noise,fourier_creation
from hypothesis import t_stat_mult_regression, t_stat

# Progress bar
toolbar_width=len(sub_list)
sys.stdout.write("GLM, :  ")
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1))


#Run GLM for each subject
for i in sub_list:
    name = i 
    behav=pd.read_table(path_to_data+name+behav_suffix,sep=" ")
    num_TR = float(behav["NumTRs"])    
    img = nib.load(smooth_data+ i +"_bold_smoothed.nii")
    data = img.get_data()    
        
        
    n_vols = data.shape[-1]    
    convolve = np.loadtxt(hrf_data+i+"_hrf.txt")
    
    residual_final = np.zeros((data.shape))
    t_final = np.zeros((data.shape[:-1]))
    p_final = np.zeros((data.shape[:-1]))
    
    #Run per slice in order to correct for time
    for j in range(data.shape[2]):
        
        data_slice = data[:,:,j,:]
        
        #Create design matrix
        X = np.ones((n_vols,6))
        X[:,1] = convolve[:,j]
        X[:,2]=np.linspace(-1,1,num=X.shape[0]) #drift
        X[:,3:]=fourier_creation(X.shape[0],3)[:,1:]
        
        beta,t,df,p = t_stat_mult_regression(data_slice, X)
        
        
        t = t[1,:]
        p = p[1,:]
        
        MRSS, fitted, residuals = glm_diagnostics(beta, X, data_slice)
        
        
        t_final[:,:,j] = t.reshape(data_slice.shape[:-1])
        p_final[:,:,j] = p.reshape(data_slice.shape[:-1])        
        residual_final[:,:,j,:] = residuals.reshape(data_slice.shape)
        
        np.save("../data/glm/t_stat/"+i+"_tstat.npy", t_final)
        np.save("../data/glm/residual/"+i+"_residual.npy", residual_final)
        np.save("../data/glm/p-values/"+i+"_pvalue.npy", p_final)
        
     
    sys.stdout.write("-")
    sys.stdout.flush()
sys.stdout.write("\n")
    
    


        
        
        
        
    
    