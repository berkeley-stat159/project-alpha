# masking_reshape_functions.py
# this file provides a way to mask data, then reduce it's dimensions 
# (and then create the correct output after analysis is done on 1d to 2d data)

def masking_reshape_start(data,mask):
	"""
	takes a 3 or 4d data and utilizes a mask to return a 1 or 2d reshaped output

	Input:
	------
	data: 3d *or* 4d np.array  (x,y,z) or (x,y,z,t) shape
	mask: a 3d np array  (x,y,z) shape, with values 0s and 1s (1 desired, 0 remove)

	Returns:
	--------
	reshaped: a 1d *or* 2d np.array (connected to 3d or 4d "data" input)

	"""
	assert(len(data.shape) == 3 or len(data.shape) == 4)

	if len(data.shape) == 3:
		data_1d= np.ravel(data)
		reshaped = data_1d[np.ravel(mask==1)]

	if len(data.shape) == 4:
		data_2d=data.reshape((-1,data.shape[-1]))
		reshaped = data_2d[np.ravel(mask==1),:]
	return reshaped



def masking_reshape_end(data_small,mask,off_value=0):
	"""
	takes a 1d input, utilizes a mask to convert into 3d output ()

	Notes:
	------
	mask must have same number of ones as the data_small.shape[0]

	Input:
	------
	data_small: a 1d np.array
	mask:       a 3d np array  (x,y,z) shape, with values 0s and 1s (1 desired, 0 remove), see notes
	off_value:  the value to be replaced for the non-on values of the mask


	Returns:
	--------
	data_big: 3d np.array  (x,y,z) shape

	"""
	assert(len(data_small.shape)==1)

	data_big = off_value*np.ones((mask.shape))

	data_big[mask==1]= data_small

	return data_big