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



from glm import glm_multiple, glm_diagnostics
# iv. import image viewing tool
from Image_Visualizing import present_3d
from noise_correction import mean_underlying_noise, fourier_predict_underlying_noise,fourier_creation
from hypothesis import t_statl

# Progress bar
toolbar_width=len(sub_list)
sys.stdout.write("Con, with 'fwhm = 1.5':  ")
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

for i in every_person:
    
    
    data = #person's smoothed data (64*64*34*120)
    
    convolve = #convolution data for the subject
    
    residual_final = np.zeros((data.shape))
    t_final = np.zeroes((data.shape[-1]))
    
    for j in range(data.shape[2])):
        
        data_slice = data[:,:,j,:]
        X = np.ones((n_vols,6))
        X[:,1] = convolve[:,j]
        X[:,2]=np.linspace(-1,1,num=X.shape[0]) #drift
        X[:,3:]=fourier_creation(X.shape[0],3)[:,1:]
        
        beta,t,df,p = t_stat(data_slice, convolve[j,:], np.array([0,1,0,0,0,0]))
        
        MRSS, fitted, residuals = glm_diagnostics(beta, X, data_slice)
        
        t_final[:,:,j] = t.reshape(data_slice.shape[:-1])
        
        residual_final[:,:,j,:] = residuals.reshape(data_slice.shape)
    
    


        
        
        
        
    
    