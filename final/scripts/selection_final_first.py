from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.linalg as npl
import nibabel as nib
import pandas as pd # new
import sys # instead of os
import os

# Relative path to subject 1 data

project_path          = "../../"
path_to_data          = project_path+"data/ds009/"
location_of_images    = project_path+"images/"
location_of_functions = project_path+"code/utils/functions/" 
final_data            = "../../final/data/"
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
from model_comparison import adjR2, BIC, AIC

# Progress bar
toolbar_width=3
sys.stdout.write("Model Selection, :  ")
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1))



for input_var in ['adjR2','BIC','AIC']:

    model1=[]
    model2=[]
    model3=[]
    model4=[]
    model4_5=[]
    model5=[]
    model6=[]
    model7=[]
    model8=[]
    model9=[]
    model9_5=[]
    model10=[]   

    if input_var == 'adjR2':
    
        def model(MRSS,y_1d,df, rank):
    	    return adjR2(MRSS,y_1d, df, rank)
        
    elif input_var == 'BIC': 
       
        def model(MRSS,y_1d,df, rank):
            return BIC(MRSS, y_1d, df, rank)
        
    elif input_var == 'AIC': 
       
        def model(MRSS,y_1d,df, rank):
            return AIC(MRSS, y_1d, df, rank)
    
    #LOAD THE DATA In
    subjects=['sub002','sub003','sub014']
    ben=0
    sys.stdout.write(subjects[ben]+": "+"[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))
    
    for i in ['sub002','sub003','sub014']:
        img = nib.load(smooth_data+ i +"_bold_smoothed.nii")
        data = img.get_data() 

        behav=pd.read_table(path_to_data+i+behav_suffix,sep=" ")
        num_TR = float(behav["NumTRs"])
        n_vols=num_TR

        hrf_matrix_all = np.loadtxt("../data/hrf/"+i+"_hrf_all.txt")
        hrf_matrix_1   = np.loadtxt("../data/hrf/"+i+"_hrf_1.txt")
        hrf_matrix_2   = np.loadtxt("../data/hrf/"+i+"_hrf_2.txt")
        hrf_matrix_3   = np.loadtxt("../data/hrf/"+i+"_hrf_3.txt")
  


        mask = nib.load(path_to_data+i+'/anatomy/inplane001_brain_mask.nii.gz')
        mask_data = mask.get_data()
        mask_data = make_mask(np.ones(data.shape[:-1]), mask_data, fit=True)
        mask_data = mask_data!=0
        mask_data = mask_data.astype(int)

        ###PCA SHIT###

        to_2d= masking_reshape_start(data,mask_data)
        # double_centered_2d
        X_pca= to_2d - np.mean(to_2d,0) - np.mean(to_2d,1)[:,None]

        cov = X_pca.T.dot(X_pca)

        U, S, V = npl.svd(cov)
        pca_addition= U[:,:6] # ~40% coverage



        #START DOING GLM
        for j in range(data.shape[2]):

            data_slice = data[:,:,j,:]
            mask_slice = mask_data[:,:,j]
            data_slice = data_slice.reshape((-1,int(num_TR)))
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
        
            rank1 = npl.matrix_rank(X[:,0:2])
        
            count = 0

            for value in MRSS1:
                model1_slice[count] = model(value, np.array(data_slice[count,:]) ,df1, rank1)  
                count+=1

            model1=model1+model1_slice.tolist()


            ###################
            #   MODEL 2       #
            ###################

            # 1.2 hrf + drift

            beta2,t,df2,p = t_stat_mult_regression(data_slice, X[:,0:3])
        
            MRSS2, fitted, residuals = glm_diagnostics(beta2, X[:,0:3], data_slice)

            model2_slice = np.zeros(len(MRSS2))
        
            rank2 = npl.matrix_rank(X[:,0:3])
            count = 0

            for value in MRSS2:
                model2_slice[count] = model(value, np.array(data_slice[count,:]) ,df2, rank2)  
                count+=1

            model2=model2+model2_slice.tolist()


            ###################
            #   MODEL 3       #
            ###################

            # 1.3 hrf + drift + fourier

            beta3,t,df3,p = t_stat_mult_regression(data_slice, X[:,0:7])
        
            MRSS3, fitted, residuals = glm_diagnostics(beta3, X[:,0:7], data_slice)

            model3_slice = np.zeros(len(MRSS3))
        
            rank3 = npl.matrix_rank(X[:,0:7])
            count = 0

            for value in MRSS3:
                model3_slice[count] = model(value, np.array(data_slice[count,:]) ,df3, rank3)  
                count+=1

            model3=model3+model3_slice.tolist()

            ###################
            #   MODEL 4       #
            ###################

            # 1.4 hrf + drift + pca

            beta4,t,df4,p = t_stat_mult_regression(data_slice, X[:,[0,1,2,7,8,9,10,11,12]])
        
            MRSS4, fitted, residuals = glm_diagnostics(beta4, X[:,[0,1,2,7,8,9,10,11,12]], data_slice)

            model4_slice = np.zeros(len(MRSS4))
            rank4 = npl.matrix_rank(X[:,[0,1,2,7,8,9,10,11,12]])
            count = 0

            for value in MRSS4:
                model4_slice[count] = model(value, np.array(data_slice[count,:]) ,df4, rank4)  
                count+=1

            model4=model4+model4_slice.tolist()

            ###################
            #   MODEL 4_5       #
            ###################

            # 1.4 hrf + drift + pca

            beta4_5,t,df4_5,p = t_stat_mult_regression(data_slice, X[:,[0,1,2,7,8,9,10]])
        
            MRSS4_5, fitted, residuals = glm_diagnostics(beta4_5, X[:,[0,1,2,7,8,9,10]], data_slice)

            model4_5_slice = np.zeros(len(MRSS4_5))
            rank4_5 = npl.matrix_rank(X[:,[0,1,2,7,8,9,10,11,12]])
            count = 0

            for value in MRSS4_5:
                model4_5_slice[count] = model(value, np.array(data_slice[count,:]) ,df4_5, rank4_5)  
                count+=1

            model4_5=model4_5+model4_5_slice.tolist()
            ###################
            #   MODEL 5       #
            ###################

            # 1.5 hrf + drift + pca + fourier

            beta5,t,df5,p = t_stat_mult_regression(data_slice, X)
        
            MRSS5, fitted, residuals = glm_diagnostics(beta5, X, data_slice)

            model5_slice = np.zeros(len(MRSS5))
        
            rank5 = npl.matrix_rank(X)
        
            count = 0

            for value in MRSS5:
                model5_slice[count] = model(value, np.array(data_slice[count,:]) ,df5, rank5)  
                count+=1

            model5=model5+model5_slice.tolist()


            ###################
            #   MODEL 6       #
            ###################

            # 2.1 hrf

            beta6,t,df6,p = t_stat_mult_regression(data_slice, X_cond[:,0:4])

            MRSS6, fitted, residuals = glm_diagnostics(beta6, X_cond[:,0:4], data_slice)

            model6_slice = np.zeros(len(MRSS6))
        
            rank6 = npl.matrix_rank(X_cond[:,0:4])
        
            count = 0

            for value in MRSS6:
                model6_slice[count] = model(value, np.array(data_slice[count,:]) ,df6, rank6)
                count+=1

            model6=model6+model6_slice.tolist()

            ###################
            #   MODEL 7       #
            ###################

            # 2.2 hrf + drift

            beta7,t,df7,p = t_stat_mult_regression(data_slice, X_cond[:,0:5])

            MRSS7, fitted, residuals = glm_diagnostics(beta7, X_cond[:,0:5], data_slice)

            model7_slice = np.zeros(len(MRSS7))
        
            rank7 = npl.matrix_rank(X_cond[:,0:5])
        
            count = 0

            for value in MRSS7:
                model7_slice[count] = model(value, np.array(data_slice[count,:]) ,df7, rank7)
                count+=1

            model7=model7+model7_slice.tolist()

            ###################
            #   MODEL 8       #
            ###################

            # 2.3 hrf + drift + fourier


            beta8,t,df8,p = t_stat_mult_regression(data_slice, X_cond[:,0:9])

            MRSS8, fitted, residuals = glm_diagnostics(beta8, X_cond[:,0:9], data_slice)

            model8_slice = np.zeros(len(MRSS8))
        
            rank8 = npl.matrix_rank(X_cond[:,0:9])
        
            count = 0

            for value in MRSS8:
                model8_slice[count] = model(value, np.array(data_slice[count,:]) ,df8, rank8)
                count+=1

            model8=model8+model8_slice.tolist()

            ###################
            #   MODEL 9       #
            ###################

            # 2.4 hrf + drift + pca


            beta9,t,df9,p = t_stat_mult_regression(data_slice, X_cond[:,list(range(5))+list(range(9,15))])

            MRSS9, fitted, residuals = glm_diagnostics(beta9,X_cond[:,list(range(5))+list(range(9,15))], data_slice)

            model9_slice = np.zeros(len(MRSS9))
        
            rank9 = npl.matrix_rank(X_cond[:,list(range(5))+list(range(9,15))])
        
            count = 0

            for value in MRSS9:
                model9_slice[count] = model(value, np.array(data_slice[count,:]) ,df9, rank9)
                count+=1

            model9=model9+model9_slice.tolist()

            ###################
            #   MODEL 9_5       #
            ###################

            # 2.4 hrf + drift + pca


            beta9_5,t,df9_5,p = t_stat_mult_regression(data_slice, X_cond[:,list(range(5))+list(range(9,13))])

            MRSS9_5, fitted, residuals = glm_diagnostics(beta9_5,X_cond[:,list(range(5))+list(range(9,13))], data_slice)

            model9_5_slice = np.zeros(len(MRSS9_5))
        
            rank9_5 = npl.matrix_rank(X_cond[:,list(range(5))+list(range(9,13))])
        
            count = 0

            for value in MRSS9_5:
                model9_5_slice[count] = model(value, np.array(data_slice[count,:]) ,df9_5, rank9_5)
                count+=1

            model9_5=model9_5+model9_5_slice.tolist()

            ###################
            #   MODEL 10       #
            ###################

            # 2.5 hrf + drift + pca + fourier
            beta10,t,df10,p = t_stat_mult_regression(data_slice, X_cond)

            MRSS10, fitted, residuals = glm_diagnostics(beta10, X_cond, data_slice)

            model10_slice = np.zeros(len(MRSS10))
        
            rank10 = npl.matrix_rank(X_cond)
        
            count = 0

            for value in MRSS10:
                model10_slice[count] = model(value, np.array(data_slice[count,:]) ,df10, rank10)
                count+=1
            model10=model10+model10_slice.tolist()



            sys.stdout.write("-")
            sys.stdout.flush()
    ben=ben+1
    sys.stdout.write("\n")
    sys.stdout.write(subjects[ben]+": "+"[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))

    final = np.array([np.mean(model1), np.mean(model2), np.mean(model3), np.mean(model4), np.mean(model4_5), np.mean(model5), np.mean(model6), np.mean(model7),
    np.mean(model8), np.mean(model9),np.mean(model9_5), np.mean(model10)])   

    final = final.reshape((2,6))   

    np.savetxt('../data/model_comparison/'+ input_var+'.txt', final)
    
    sys.stdout.write("-")
    sys.stdout.flush()
sys.stdout.write("\n")



import matplotlib.pyplot as plt

aic=np.loadtxt("../data/model_comparison/AIC.txt")
bic=np.loadtxt("../data/model_comparison/BIC.txt")
adjR2=np.loadtxt("../data/model_comparison/adjR2.txt")

plt.plot([1,2,3,4,4.5,5],aic[0,:],label="all conditions together")
plt.plot([1,2,3,4,4.5,5],aic[1,:],label="individual conditions")
plt.title("AIC")
plt.legend(loc='top right', shadow=True,fontsize="smaller")
plt.savefig('../../images/aic.png')
plt.close()

plt.plot([1,2,3,4,4.5,5],bic[0,:],label="all conditions together")
plt.plot([1,2,3,4,4.5,5],bic[1,:],label="individual conditions")
plt.title("BIC")
plt.legend(loc='top right', shadow=True,fontsize="smaller")
plt.savefig('../../images/bic.png')
plt.close()

plt.plot([1,2,3,4,4.5,5],adjR2[0,:],label="all conditions conditions")
plt.plot([1,2,3,4,4.5,5],adjR2[1,:],label="individual conditions")
plt.title("Adjusted R2")
plt.legend(loc='top right', shadow=True,fontsize="smaller")
plt.savefig('../../images/adjr2.png')
plt.close()

np.round(aic,3)
np.round(bic,3)
np.round(adjR2,3)







aic=np.loadtxt("../data/model_comparison/AIC.txt")
bic=np.loadtxt("../data/model_comparison/BIC.txt")
adjR2=np.loadtxt("../data/model_comparison/adjR2.txt")

# making the plots make more sense:
aic_better=aic.copy()
bic_better=bic.copy()
adjR2_better=adjR2.copy()
aic_better[:,3],aic_better[:,4]     = aic[:,4],aic[:,3]
bic_better[:,3],bic_better[:,4]     = bic[:,4],bic[:,3]
adjR2_better[:,3],adjR2_better[:,4] = adjR2[:,4],adjR2[:,3]

plt.plot(np.arange(6)+1,aic_better[0,:],label="all conditions together")
plt.plot(np.arange(6)+1,aic_better[1,:],label="individual conditions")
plt.title("AIC")
plt.legend(loc='top right', shadow=True,fontsize="smaller")
plt.savefig('../../images/aic_better.png')
plt.close()

plt.plot(np.arange(6)+1,bic_better[0,:],label="all conditions together")
plt.plot(np.arange(6)+1,bic_better[1,:],label="individual conditions")
plt.title("BIC")
plt.legend(loc='top right', shadow=True,fontsize="smaller")
plt.savefig('../../images/bic_better.png')
plt.close()

plt.plot(np.arange(6)+1,adjR2_better[0,:],label="all conditions conditions")
plt.plot(np.arange(6)+1,adjR2_better[1,:],label="individual conditions")
plt.title("Adjusted R2")
plt.legend(loc='top right', shadow=True,fontsize="smaller")
plt.savefig('../../images/adjr2_better.png')
plt.close()

