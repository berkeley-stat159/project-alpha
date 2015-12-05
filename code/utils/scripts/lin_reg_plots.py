# plots for paper:
# from glm_final.py

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

# Relative path to subject 1 data

project_path          = "../../../"
path_to_data          = project_path+"data/ds009/"
location_of_images    = project_path+"paper/images/"
location_of_functions = project_path+"code/utils/functions/" 
final_data            = project_path+"final/data/"
smooth_data           = final_data + 'smooth/'
hrf_data              = final_data + 'hrf/'


#sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
sys.path.append(location_of_functions)

sub_list = os.listdir(path_to_data)[1:]

from glm import glm_multiple, glm_diagnostics
# iv. import image viewing tool
from Image_Visualizing import present_3d
from noise_correction import mean_underlying_noise, fourier_predict_underlying_noise,fourier_creation
from hypothesis import t_stat_mult_regression
# Progress bar

i=sub_list[0]
j=15

img = nib.load(smooth_data+ i +"_bold_smoothed.nii")
data = img.get_data().astype(float)
    
n_vols = data.shape[-1]    
convolve = np.loadtxt(hrf_data+i+"_hrf.txt")

residual_final = np.zeros((data.shape))
t_final = np.zeros((data.shape[:-1]))


data_slice = data[:,:,j,:]
        
X = np.ones((n_vols,7))
X[:,1] = convolve[:,j]
X[:,2]=np.linspace(-1,1,num=X.shape[0]) #drift
X[:,3:]=fourier_creation(X.shape[0],2)[:,1:]

beta,t,df,p = t_stat_mult_regression(data_slice, X)


MRSS, fitted, residuals = glm_diagnostics(beta, X, data_slice)




plt.scatter(fitted[30,40,:],residuals[30,40,:])
plt.title("Subject 001, voxel: [30,40,15]")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.savefig(location_of_images+'Fitted_v_Residuals.png')
plt.close()



plt.plot(fitted[30,40,:],label="fitted")
plt.plot(data_slice[30,40],label="recorded fMRI")
plt.title("Subject 001, voxel: [30,40,15]")
plt.xlabel("Time Course")
plt.ylabel("Brain fMRI recording")
plt.legend(loc='lower right', shadow=True,fontsize="smaller")
plt.savefig(location_of_images+'Fitted_v_Actual.png')
plt.close()





convolve_all = np.loadtxt(hrf_data+i+"_hrf.txt")[:,0]
convolve_1 = np.loadtxt(hrf_data+i+"_hrf_1.txt")[:,0]
convolve_2 = np.loadtxt(hrf_data+i+"_hrf_2.txt")[:,0]
convolve_3 = np.loadtxt(hrf_data+i+"_hrf_3.txt")[:,0]


xx=np.arange(len(convolve_all))*2
plt.plot(xx,convolve_all,color="#4000ff",label="All Conditions")
plt.plot(xx[[0,len(xx)-1]],[0,0],color="k")
plt.plot(xx,convolve_1-2,color="#0070ff",label="Condition 1")
plt.plot(xx[[0,len(xx)-1]],[-2,-2],color="k")
plt.plot(xx,convolve_2-4,color="#00bfff",label="Condition 2")
plt.plot(xx[[0,len(xx)-1]],[-4,-4],color="k")
plt.plot(xx,convolve_3-6,color="#00ffff",label="Condition 3")
plt.plot(xx[[0,len(xx)-1]],[-6,-6],color="k")
plt.title("Subject 001, first slice")
plt.xlabel("Time Course")
plt.yticks([])
plt.ylabel("Convolution")
plt.legend(loc='center right', shadow=True,fontsize="smaller")
plt.savefig(location_of_images+'all_cond_time.png')
plt.close()

