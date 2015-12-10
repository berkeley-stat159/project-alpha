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
behav_suffix           = "/behav/task001_run001/behavdata.txt"
smooth_data           =  final_data + 'smooth/'
hrf_data              = final_data + 'hrf/'
residual_data              = final_data + 'glm/residual/'

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
sys.stdout.write("GLM, :  ")
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

# Set up lists to store proportion of p-values above 0.05. 
unmasked_prop = [] # Unmasked (all voxels)
masked_prop = [] # Masked. 

for i in sub_list:
    residuals =   np.load(residual_data+i+"_residual.npy")
    sw_pvals = check_sw(residuals)
    unmasked_prop.append(np.mean(sw_pvals > 0.05))

    mask = nib.load(path_to_data+i+'/anatomy/inplane001_brain_mask.nii.gz')
    mask_data = mask.get_data()
    
    masked_pvals = make_mask(sw_pvals, mask_data, fit=True)
    pvals_in_brain = sw_pvals.ravel()[masked_pvals.ravel() != 0]
    masked_prop.append(np.mean(pvals_in_brain > 0.05))
     
    sys.stdout.write("-")
    sys.stdout.flush()
sys.stdout.write("\n")

print("Average proportion of unmasked p-values above 0.05:" + str(np.mean(np.array(unmasked_prop))))
print("Average proportion of masked p-values above 0.05:" + str(np.mean(np.array(masked_prop))))

plt.close()
plt.hist(np.array(masked_prop))
plt.title("Histogram of Proportions for each Subject")
plt.savefig(location_of_images+'maskedhist.png')
plt.close()

# Save image plots of unmasked p-values for a single subject. 
plt.imshow(present_3d(sw_pvals), cmap=plt.get_cmap('gray'))
plt.savefig(location_of_images+i+'sw.png')
plt.close()

# Save image plots of masked p-values for a single subject. 
plt.imshow(present_3d(masked_pvals), cmap=plt.get_cmap('gray'))
plt.savefig(location_of_images+i+'swmasked.png')
plt.close()


    
    


        
        
        
        
    
    
