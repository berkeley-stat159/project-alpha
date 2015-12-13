# visualizing the brain's other cuts

# assumed in final:

from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd # new
import sys # instead of os
import os


# Relative path to subject all of the subjects

project_path          = "../../../"
path_to_data          = project_path+"data/ds009/"
location_of_images    = project_path+"images/"
location_of_functions = project_path+"code/utils/functions/" 
final_data            = "../data/"
behav_suffix           = "/behav/task001_run001/behavdata.txt"
smooth_data           =  final_data + 'smooth/'
hrf_data              = final_data + 'hrf/'

sys.path.append(location_of_functions)
from Image_Visualizing import present_3d, make_mask, present_3d_options

i="sub001"
brain=nib.load(path_to_data+i+'/BOLD/task001_run001/bold.nii.gz')
data=brain.get_data()

brain_hi= nib.load(path_to_data + i+'/anatomy/inplane001_brain.nii.gz')
joy_hi=brain_hi.get_data()

joy=data[...,7]
plt.imshow(present_3d(data[...,7]),cmap="gray",interpolation="nearest")



plt.figure()
plt.imshow(present_3d_options(joy,2),cmap="gray",interpolation="nearest")
plt.title("2")
plt.figure()
plt.imshow(present_3d_options(joy,1),cmap="gray",interpolation="nearest")
plt.title("1")
plt.figure()
plt.imshow(present_3d_options(joy,0),cmap="gray",interpolation="nearest")
plt.title("0")




plt.close()
plt.imshow(present_3d_options(joy_hi,2),cmap="gray",interpolation="nearest")
plt.title("2")
plt.savefig(location_of_images+"kent_brain_2.png")
plt.close()

plt.imshow(present_3d_options(joy_hi,1),cmap="gray",interpolation="nearest")
plt.title("1")
plt.savefig(location_of_images+"kent_brain_1.png")
plt.close()


plt.imshow(present_3d_options(joy_hi,0),cmap="gray",interpolation="nearest")
plt.title("0")
plt.savefig(location_of_images+"kent_brain_0.png")
plt.close()


upper=np.percentile(np.ravel(joy_hi),95)

plt.close()
plt.imshow(present_3d_options(joy_hi[::2,::2,:],2),cmap="gray",interpolation="nearest")
plt.colorbar()
plt.clim(0,upper)
plt.title("2")
plt.savefig(location_of_images+"kent_brain_2_smaller.png")
plt.close()

plt.imshow(present_3d_options(joy_hi[::2,::2,:],1),cmap="gray",interpolation="nearest")
plt.title("1")
plt.colorbar()
plt.clim(0,upper)
plt.savefig(location_of_images+"kent_brain_1_smaller.png")
plt.close()


plt.imshow(present_3d_options(joy_hi[::2,::2,:],0),cmap="gray",interpolation="nearest")
plt.title("0")
plt.colorbar()
plt.clim(0,upper)
plt.savefig(location_of_images+"kent_brain_0_smaller.png")
plt.close()