"""
Final script to test the differences in t-values caused by smoothing and 
simple regression vs. full model. 

Design matrix takes into account conditions, drift, and pca. Also runs slice 
by slice in order to correct for time. Creates plots that shows how t-values 
change based on different models.

There are four different models we are going to test:
1. Smooth Data with full regression model
2. Un-smooth data with full regression model
3. Smooth data with simple regression model
4. Un-smooth data with simple regression model

"""

from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.linalg as npl
import nibabel as nib
import pandas as pd
import sys
import os


# Relative path to subject all of the subjects

project_path          = "../../"
path_to_data          = project_path + "data/ds009/"
location_of_images    = project_path + "images/"
location_of_functions = project_path + "code/utils/functions/" 
final_data            = "../data/"
behav_suffix           = "/behav/task001_run001/behavdata.txt"
smooth_data           =  final_data + 'smooth/'
hrf_data              = final_data + 'hrf/'


sys.path.append(location_of_functions)

sub_list = os.listdir(path_to_data)
sub_list = [i for i in sub_list if 'sub' in i]


#Import our functions
from glm import glm_multiple, glm_diagnostics
from Image_Visualizing import present_3d
from noise_correction import mean_underlying_noise, fourier_predict_underlying_noise,fourier_creation
from hypothesis import t_stat_mult_regression, t_stat

# Progress bar
toolbar_width = len(sub_list)
sys.stdout.write("GLM, :  ")
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width + 1))


#Run GLM for each subject
for i in sub_list:
    name = i 
    behav = pd.read_table(path_to_data + name + behav_suffix, sep = " ")
    num_TR = float(behav["NumTRs"])    
    img_smooth = nib.load(smooth_data + i + "_bold_smoothed.nii")
    img_rough = nib.load(path_to_data + name + "/BOLD/task001_run001/bold.nii.gz")
    
    data_smooth = img_smooth.get_data()    
    data_rough = img_rough.get_data()
    
    first_n_vols = data_rough.shape[-1]
    num_TR_cut = int(first_n_vols - num_TR)
    
    data_rough = data_rough[..., num_TR_cut:] 
    
        
    n_vols = data_smooth.shape[-1]    
    convolve = np.loadtxt(hrf_data + i + "_hrf_all.txt")
    
    t_final1 = np.zeros((data_smooth.shape[:-1]))
    t_final2 = np.zeros((data_smooth.shape[:-1]))
    t_final3 = np.zeros((data_smooth.shape[:-1]))
    
    #Run per slice in order to correct for time
    for j in range(data_smooth.shape[2]):
        
        data_smooth_slice = data_smooth[:, :, j, :]
        data_rough_slice = data_rough[:, :, j, :]    

        #Create design matrix
        X = np.ones((n_vols,7))
        X[:, 1] = convolve[:, j]
        X[:, 2] = np.linspace(-1, 1, num = X.shape[0])
        X[:, 3:] = fourier_creation(X.shape[0], 2)[:, 1:]
        
        beta1,t1,df1,p1 = t_stat_mult_regression(data_rough_slice, X)
        beta2, t2, df2, p2 = t_stat(data_smooth_slice, convolve[:, j], c = [0, 1])
        beta3, t3, df3, p3 = t_stat(data_rough_slice, convolve[:, j], c = [0, 1])
        
        
        
        t1 = t1[1, :]
        t2 = t2.T
        t3 = t3.T
                
        
        t_final1[:, :, j] = t1.reshape(data_rough_slice.shape[:-1])
        t_final2[:, :, j] = t2.reshape(data_smooth_slice.shape[:-1])
        t_final3[:, :, j] = t3.reshape(data_rough_slice.shape[:-1])
        
        
    np.save("../data/t_stat/" + i + "_tstat_rough_full.npy", t_final1)
    np.save("../data/t_stat/" + i + "_tstat_smooth_simple.npy", t_final2)
    np.save("../data/t_stat/" + i + "_tstat_rough_simple.npy", t_final3)
                
     
    sys.stdout.write("-")
    sys.stdout.flush()
sys.stdout.write("\n")
    
    


        
        
        
        
    
    