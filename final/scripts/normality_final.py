from __future__ import absolute_import, division, print_function
import numpy as np
import nibabel as nib
from scipy.stats import shapiro
import os
import sys

# Relative path to subject 1 data

project_path          = "../../"
path_to_data          = project_path+"data/ds009/"
location_of_images    = project_path+"images/"
location_of_functions = project_path+"code/utils/functions/" 
final_data            = "../data/"
behav_suffix           = "/behav/task001_run001/behavdata.txt"
smooth_data           =  final_data + 'smooth/'
hrf_data              = final_data + 'hrf/'
residual_data              = final_data + 'glm/residual/'



#sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
sys.path.append(location_of_functions)

sub_list = os.listdir(path_to_data)[1:]

from stimuli import events2neural
from event_related_fMRI_functions import hrf_single, convolution_specialized

# Load our GLM functions. 
from glm import glm, glm_diagnostics

# Load our normality functions. 
from normality import check_sw

# Progress bar
toolbar_width=len(sub_list)
sys.stdout.write("GLM, :  ")
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

for i in sub_list:
    residuals =   np.load(residual_data+i+"_residual.npy")
    sw_pvals = check_sw(residuals)
    print(np.mean(sw_pvals > 0.05))

     
    sys.stdout.write("-")
    sys.stdout.flush()
sys.stdout.write("\n")
    
    


        
        
        
        
    
    