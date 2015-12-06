from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.linalg as npl
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
location_of_images    = project_path+"images/"
location_of_functions = project_path+"code/utils/functions/" 
final_data            = "../../../final/data/"
smooth_data           =  final_data + 'smooth/'
hrf_data              = final_data + 'hrf/'
behav_suffix           = "/behav/task001_run001/behavdata.txt"


sys.path.append(location_of_functions)

sub_list = os.listdir(path_to_data)[1:]

from event_related_fMRI_functions import hrf_single, np_convolve_30_cuts
from time_shift import time_shift, make_shift_matrix, time_correct
from glm import glm_multiple, glm_diagnostics
from noise_correction import mean_underlying_noise, fourier_predict_underlying_noise,fourier_creation
from hypothesis import t_stat_mult_regression, t_stat
from Image_Visualizing import present_3d, make_mask
from mask_phase_2_dimension_change import masking_reshape_start, masking_reshape_end

input_var = input("adjR2 or BIC: ")


################
# Adjusted R^2 #
################

if input_var == 'adjR2':
    
    def model(MRSS,y_1d,df):
	    n=y_1d.shape[0]
	    RSS= MRSS*df
	    TSS= np.sum((y_1d-np.mean(y_1d))**2)
	    adjR2 = 1- ((RSS/TSS)  * (df/(n-1))  )
	    return adjR2
        
elif input_var == 'BIC': 
       
    def model(MRSS,y_1d,df):
	    n=y_1d.shape[0]
	    RSS= MRSS*df
	    BIC= n * np.log(RSS/n) + np.log(n)*(n-df)
	    return BIC
        
model1=[]
model2=[]
model3=[]
model4=[]
model5=[]
model6=[]
model7=[]
model8=[]
model9=[]
model10=[]   
    
#LOAD THE DATA In
for i in ['sub001','sub004','sub010']:
    img = nib.load(smooth_data+ i +"_bold_smoothed.nii")
    data = img.get_data() 

    behav=pd.read_table(path_to_data+i+behav_suffix,sep=" ")
    num_TR = float(behav["NumTRs"])


    #CREATE THE CONVOLVE STUFF
    cond1=np.loadtxt(path_to_data+ i+ "/model/model001/onsets/task001_run001/cond001.txt")
    cond2=np.loadtxt(path_to_data+ i+ "/model/model001/onsets/task001_run001/cond002.txt")
    cond3=np.loadtxt(path_to_data+ i+ "/model/model001/onsets/task001_run001/cond003.txt")

    TR = 2
    tr_times = np.arange(0, 30, TR)
    hrf_at_trs = np.array([hrf_single(x) for x in tr_times])

    # creating the .txt file for the events2neural function
    cond_all=np.row_stack((cond1,cond2,cond3))
    cond_all=sorted(cond_all,key= lambda x:x[0])

    cond_all=np.array(cond_all)[:,0]

    delta_y=2*(np.arange(34))/34

    shifted_all=make_shift_matrix(cond_all,delta_y)
    shifted_1= make_shift_matrix(cond1[:,0],delta_y)
    shifted_2= make_shift_matrix(cond2[:,0],delta_y)
    shifted_3= make_shift_matrix(cond3[:,0],delta_y)


    def make_convolve_lambda(hrf_function,TR,num_TRs):
        convolve_lambda=lambda x: np_convolve_30_cuts(x,np.ones(x.shape[0]),hrf_function,TR,np.linspace(0,(num_TRs-1)*TR,num_TRs),15)[0]
    
        return convolve_lambda
    
    convolve_lambda=make_convolve_lambda(hrf_single,TR,num_TR)

    hrf_matrix_all=time_correct(convolve_lambda,shifted_all,num_TR)
    hrf_matrix_1=time_correct(convolve_lambda,shifted_1,num_TR)
    hrf_matrix_2=time_correct(convolve_lambda,shifted_2,num_TR)
    hrf_matrix_3=time_correct(convolve_lambda,shifted_3,num_TR)

    n_vols = data.shape[-1]    


    mask = nib.load(path_to_data+i+'/anatomy/inplane001_brain_mask.nii.gz')
    mask_data = mask.get_data()
    mask_data = make_mask(np.ones(data.shape[:-1]), mask_data, fit=True)
    mask_data = mask_data!=0
    mask_data = mask_data.astype(int)

    ###PCA SHIT###

    to_2d= masking_reshape_start(data,mask)
    # double_centered_2d
    X_pca= to_2d - np.mean(to_2d,0) - np.mean(to_2d,1)[:,None]

    cov = X_pca.T.dot(X_pca)

    U, S, V = npl.svd(cov)
    pca_addition= U[:,:6] # ~40% coverage



    #START DOING GLM
    for j in range(data.shape[2]):

        data_slice = data[:,:,j,:]
        mask_slice = mask_data[:,:,j]
        data_slice = data_slice.reshape((-1,num_TR))
        mask_slice = np.ravel(mask_slice)

        data_slice = data_slice[mask_slice==1]

        # all conditions in 1 roof (cond_all)
        X = np.ones((n_vols,13))
        X[:,1] = hrf_matrix_all[:,j] # 1 more
        X[:,2] = np.linspace(-1,1,num=X.shape[0]) #drift # one 
        X[:,3:7] = fourier_creation(X.shape[0],2)[:,1:] # four more
        X[:,7:] = pca_addition

        # all conditions seperate (cond1,cond2,cond3)
        X_cond = np.ones((n_vols,15))
        X_cond[:,1] = hrf_matrix_1[:,j] # 1 more
        X_cond[:,2] = hrf_matrix_2[:,j] # 1 more
        X_cond[:,3] = hrf_matrix_3[:,j] # 1 more
        X_cond[:,4] = np.linspace(-1,1,num=X.shape[0]) #drift # one 
        X_cond[:,5:9] = fourier_creation(X.shape[0],2)[:,1:] # four more
        X_cond[:,9:] = pca_addition


    #START CREATING MODELS

        ###################
        #   MODEL 1       #
        ###################
        # 1.1 hrf (simple)

        beta1,t,df1,p = t_stat_mult_regression(data_slice, X[:,0:2])
        
        MRSS1, fitted, residuals = glm_diagnostics(beta1, X[:,0:2], data_slice)

        model1_slice = np.zeros(len(MRSS1))
        count = 0

        for value in MRSS1:
            model1_slice[count] = model(value, np.array(data_slice[count,:]) ,df1)  
            count+=1

        model1=model1+model1_slice.tolist()

        ###################
        #   MODEL 2       #
        ###################

        # 1.2 hrf + drift

        beta2,t,df2,p = t_stat_mult_regression(data_slice, X[:,0:3])
        
        MRSS2, fitted, residuals = glm_diagnostics(beta2, X[:,0:3], data_slice)

        model2_slice = np.zeros(len(MRSS2))
        count = 0

        for value in MRSS2:
            model2_slice[count] = model(value, np.array(data_slice[count,:]) ,df2)  
            count+=1

        model2=model2+model2_slice.tolist()

        ###################
        #   MODEL 3       #
        ###################

        # 1.3 hrf + drift + fourier

        beta3,t,df3,p = t_stat_mult_regression(data_slice, X[:,0:7])
        
        MRSS3, fitted, residuals = glm_diagnostics(beta3, X[:,0:7], data_slice)

        model3_slice = np.zeros(len(MRSS3))
        count = 0

        for value in MRSS3:
            model3_slice[count] = model(value, np.array(data_slice[count,:]) ,df3)  
            count+=1

        model3=model3+model3_slice.tolist()

        ###################
        #   MODEL 4       #
        ###################

        # 1.4 hrf + drift + pca

        beta4,t,df4,p = t_stat_mult_regression(data_slice, X[:,[0,1,2,7,8,9,10,11,12]])
        
        MRSS4, fitted, residuals = glm_diagnostics(beta4, X[:,[0,1,2,7,8,9,10,11,12]], data_slice)

        model4_slice = np.zeros(len(MRSS4))
        count = 0

        for value in MRSS4:
            model4_slice[count] = model(value, np.array(data_slice[count,:]) ,df4)  
            count+=1

        model4=model4+model4_slice.tolist()

        ###################
        #   MODEL 5       #
        ###################

        # 1.5 hrf + drift + pca + fourier

        beta5,t,df5,p = t_stat_mult_regression(data_slice, X)
        
        MRSS5, fitted, residuals = glm_diagnostics(beta5, X, data_slice)

        model5_slice = np.zeros(len(MRSS5))
        count = 0

        for value in MRSS5:
            model5_slice[count] = model(value, np.array(data_slice[count,:]) ,df5)  
            count+=1

        model5=model5+model5_slice.tolist()


        ###################
        #   MODEL 6       #
        ###################

        # 2.1 hrf

        beta6,t,df6,p = t_stat_mult_regression(data_slice, X_cond[:,0:4])

        MRSS6, fitted, residuals = glm_diagnostics(beta6, X_cond[:,0:4], data_slice)

        model6_slice = np.zeros(len(MRSS6))
        count = 0

        for value in MRSS6:
            model6_slice[count] = model(value, np.array(data_slice[count,:]) ,df6)
            count+=1

        model6=model6+model6_slice.tolist()

        ###################
        #   MODEL 7       #
        ###################

        # 2.2 hrf + drift

        beta7,t,df7,p = t_stat_mult_regression(data_slice, X_cond[:,0:5])

        MRSS7, fitted, residuals = glm_diagnostics(beta7, X_cond[:,0:5], data_slice)

        model7_slice = np.zeros(len(MRSS7))
        count = 0

        for value in MRSS7:
            model7_slice[count] = model(value, np.array(data_slice[count,:]) ,df7)
            count+=1

        model7=model7+model7_slice.tolist()

        ###################
        #   MODEL 8       #
        ###################

        # 2.3 hrf + drift + fourier


        beta8,t,df8,p = t_stat_mult_regression(data_slice, X_cond[:,0:9])

        MRSS8, fitted, residuals = glm_diagnostics(beta8, X_cond[:,0:9], data_slice)

        model8_slice = np.zeros(len(MRSS8))
        count = 0

        for value in MRSS8:
            model8_slice[count] = model(value, np.array(data_slice[count,:]) ,df8)
            count+=1

        model8=model8+model8_slice.tolist()

        ###################
        #   MODEL 9       #
        ###################

        # 2.4 hrf + drift + pca


        beta9,t,df9,p = t_stat_mult_regression(data_slice, X_cond[:,list(range(5))+list(range(9,15))])

        MRSS9, fitted, residuals = glm_diagnostics(beta9,X_cond[:,list(range(5))+list(range(9,15))], data_slice)

        model9_slice = np.zeros(len(MRSS9))
        count = 0

        for value in MRSS9:
            model9_slice[count] = model(value, np.array(data_slice[count,:]) ,df9)
            count+=1

        model9=model9+model9_slice.tolist()


        ###################
        #   MODEL 10       #
        ###################

        # 2.5 hrf + drift + pca + fourier
        beta10,t,df10,p = t_stat_mult_regression(data_slice, X_cond)

        MRSS10, fitted, residuals = glm_diagnostics(beta10, X_cond, data_slice)

        model10_slice = np.zeros(len(MRSS10))
        count = 0

        for value in MRSS10:
            model10_slice[count] = model(value, np.array(data_slice[count,:]) ,df10)
            count+=1

        model10=model10+model10_slice.tolist()

    
final = np.array([np.mean(model1), np.mean(model2), np.mean(model3), np.mean(model4), np.mean(model5), np.mean(model6), np.mean(model7),
np.mean(model8), np.mean(model9), np.mean(model10)])   

final = final.reshape((2,5))   
###################
# Desired Models: #
###################

### subcategory 1:
# with only cond_all hrf run

# 1.1 hrf (simple)
# 1.2 hrf + drift
# 1.3 hrf + drift + fourier
# 1.4 hrf + drift + pca
# 1.5 hrf + drift + pca += fourier




### subcategory 1:
# with all 3 different hrfs for each type of condition

# 2.1 hrf
# 2.2 hrf + drift
# 2.3 hrf + drift + fourier
# 2.4 hrf + drift + pca
# 2.5 hrf + drift + pca + fourier



# need to correct fourier in noise correction function file