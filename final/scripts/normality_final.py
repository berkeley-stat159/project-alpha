"""
Script for Normality

Runs the Shapiro-Wilks Test for Normality on residuals

Compares unmasked vs masked data and plots for one subject

"""


from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.stats import shapiro
import os
import sys

# Relative path to all of the subjects

project_path          = "../../"
path_to_data          = project_path+"data/ds009/"
location_of_images    = project_path+"images/"
location_of_functions = project_path+"code/utils/functions/" 
final_data            = "../data/"
behav_suffix          = "/behav/task001_run001/behavdata.txt"
smooth_data           =  final_data + 'smooth/'
hrf_data              = final_data + 'hrf/'
residual_data         = final_data + 'residual/'

sys.path.append(location_of_functions)

sub_list = os.listdir(path_to_data)
sub_list = [i for i in sub_list if 'sub' in i]


from stimuli import events2neural
from event_related_fMRI_functions import hrf_single, convolution_specialized

# Load our GLM functions. 
from glm import glm, glm_diagnostics

# Load our normality functions. 
from normality import check_sw

# Load masking and visualization functions.
from Image_Visualizing import make_mask, present_3d

# Progress bar
toolbar_width=len(sub_list)
sys.stdout.write("Shaprio-Wilk test for normality, :  ")
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

# Set up lists to store proportion of p-values above 0.05. 
unmasked_prop = [] # Unmasked (all voxels)
masked_prop = [] # Masked. 

for i in sub_list:
    residuals =   np.load(residual_data+i+"_residual_fourier.npy")
    sw_pvals = check_sw(residuals)
    unmasked_prop.append(np.mean(sw_pvals > 0.05))

    mask = nib.load(path_to_data+i+'/anatomy/inplane001_brain_mask.nii.gz')
    mask_data = mask.get_data()
    
    masked_pvals = make_mask(sw_pvals, mask_data, fit=True)
    pvals_in_brain = sw_pvals.ravel()[masked_pvals.ravel() != 0]
    masked_prop.append(np.mean(pvals_in_brain > 0.05))
    
    if (i[-3:]=="010"): 
        # Save image plots of unmasked p-values for subject 10. 
        plt.imshow(present_3d(sw_pvals), cmap=plt.get_cmap('gray'))
        plt.savefig(location_of_images+i+'sw.png')
        plt.close()

        # Save image plots of masked p-values for a single subject. 
        plt.imshow(present_3d(masked_pvals), cmap=plt.get_cmap('gray'))
        plt.savefig(location_of_images+i+'swmasked.png')
        plt.close()
     
    sys.stdout.write("-")
    sys.stdout.flush()
sys.stdout.write("\n")

print("Average proportion of p-values above 0.05 (unmasked):" + str(np.mean(np.array(unmasked_prop))))
print("Average proportion of p-values above 0.05 (masked):" + str(np.mean(np.array(masked_prop))))
    
    


        
        
        
        
    
    
