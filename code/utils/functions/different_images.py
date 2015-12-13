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
from Image_Visualizing import present_3d, make_mask

i="sub001"
brain=nib.load(path_to_data+i+'/BOLD/task001_run001/bold.nii.gz')
data=brain.get_data()

brain_hi= nib.load(path_to_data + i+'/anatomy/inplane001_brain.nii.gz')
joy_hi=brain_hi.get_data()

joy=data[...,7]
plt.imshow(present_3d(data[...,7]),cmap="gray",interpolation="nearest")


def present_3d_options(three_d_image,axis=2):
	""" Coverts a 3d image into a 2nd image with slices in 3rd dimension varying across the element
	three_d_image: is a 3 dimensional numpy array
	
	# might later add these in (couldn't do so at this time) 
	num_images: number of 2d images in 3d array (or number wanted to print out)
	image_dim: the dimension of the image

	#####
	With Results just do:
	In[0]: full=present_3d(three_d_image)
	In[1]: plt.imshow(full,cmap="gray",interpolation="nearest")
	In[2]: plt.colorbar()
	"""
	assert(axis in [0,1,2])
	assert(len(three_d_image.shape)==3)

	num_images=three_d_image.shape[axis]  

	if axis==0:
		image_dim=list(three_d_image.shape[1:])
		image_dim.reverse()
	elif axis==1:
		image_dim=(three_d_image.shape[2],three_d_image.shape[0])
	else:
		image_dim=three_d_image.shape[:2]



	# formating grid
	length=np.ceil(np.sqrt(num_images))
	grid_size=[int(x) for x in (length,np.ceil(num_images/length))]

	full=np.zeros((image_dim[0]*grid_size[0],image_dim[1]*grid_size[1]))
	counter=0


	if axis==0:
		for row in range(int(grid_size[0])):
			for col in range(int(grid_size[1])):
				if counter< num_images:
					full[(row*image_dim[0]):((row+1)*image_dim[0]),(col*image_dim[1]):((col+1)*image_dim[1])]=np.rot90(three_d_image[row*grid_size[1]+col,...],1)
				counter=counter+1
		return full
	elif axis==1:
		for row in range(int(grid_size[0])):
			for col in range(int(grid_size[1])):
					if counter< num_images:
						full[(row*image_dim[0]):((row+1)*image_dim[0]),(col*image_dim[1]):((col+1)*image_dim[1])]=np.rot90(three_d_image[:,row*grid_size[1]+col,:],2).T
					counter=counter+1
		return full
	else: # regular:
		for row in range(int(grid_size[0])):
			for col in range(int(grid_size[1])):
				if counter< num_images:
					full[(row*image_dim[0]):((row+1)*image_dim[0]),(col*image_dim[1]):((col+1)*image_dim[1])]=three_d_image[...,row*grid_size[1]+col]
				counter=counter+1
		return full

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

