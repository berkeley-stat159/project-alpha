import numpy as np

def present_3d(three_d_image):
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
    num_images=three_d_image.shape[-1]  
    image_dim=three_d_image.shape[0:2]  

    # formating grid
    length=np.ceil(np.sqrt(num_images))
    grid_size=(length,np.ceil(num_images/length))
    
    full=np.zeros((image_dim[0]*grid_size[0],image_dim[1]*grid_size[1]))
    counter=0
    
    for row in range(int(grid_size[0])):
        for col in range(int(grid_size[1])):
            if counter< num_images:
                full[(row*image_dim[0]):((row+1)*image_dim[0]),(col*image_dim[0]):((col+1)*image_dim[0])]=three_d_image[...,row*grid_size[1]+col]
            counter=counter+1
    return full


