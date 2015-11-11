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

def make_mask(data_3d, mask_data, fit=False):
    """ Takes a 3d image and a 3d mask array and fits the mask over the 3d image. 
    
    The mask turns all of the points of data_3d that are not part of the mask into 0's.
    
    If 'fit=True', then the resolution of the mask is different from the resolution 
    of the image and the array change the resolution of the mask. 
    

   Parameters
   ----------
   data_3d: numpy array of 3 dimensions 
       The image data of one subject that you wish to fit mask over
   mask_data: numpy array of 3 dimension
       The mask for the data_3d
   fit: boolean
       Whether or not the resolution of the mask needs to be altered to fit onto the data
   
   Returns
   -------
   new_data: numpy array of 3 dimensions
       Same data frame as data_3d but with the mask placed on top.
       
    """
    def shrink(data, rows, cols):
        return data.reshape(rows, data.shape[0]/rows, cols, data.shape[1]/cols).sum(axis=1).sum(axis=2)
        
    if fit == False:
        if data_3d.shape != mask_data.shape:
            raise ValueError('The shape of mask and data are not the same. Trying making "fit=True"')
        else:
            return data_3d * mask_data
    
    elif fit== True:
        new_mask = np.zeros(data_3d.shape)
        for j in range(mask_data.shape[-1]):
            new_mask[...,j] = shrink(mask_data[...,j], data_3d.shape[0], data_3d.shape[1])

        return data_3d * new_mask


