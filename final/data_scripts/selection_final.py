"""
Tests 10 different models and selects best one using AIC, BIC and Adjusted R2.

Our design matrix includes a subset of the following features:
    hrf (simple):   a single HRF for all the conditions
    hrf:            3 HRF for each condition
    drift:          linear drift correction
    fourier:        time courses' Fourier series
    pca:            principal components

Below are the models that we test:

    Model 1:        hrf (simple)
    Model 2:        hrf (simple) + drift
    Model 3:        hrf (simple)  + drift + fourier
    Model 4:        hrf (simple)  + drift + pca 6
    Model 4.5:      hrf (simple)  + drift + pca 4
    Model 5:        hrf (simple)  + drift + pca 6 + fourier
    Model 6:        hrf
    Model 7:        hrf + drift
    Model 8:        hrf + drift + fourier
    Model 9:        hrf + drift + pca 6
    Model 9.5:      hrf + drift + pca 4
    Model 10:       hrf + drift + pca 6 + fourier

Ultimately, we chose Model 4 to be the best model.

"""

from __future__ import absolute_import, division, print_function
import numpy as np
import numpy.linalg as npl
import nibabel as nib
import pandas as pd
import sys
import os

# Relative path to subject 1 data

project_path          = "../../"
path_to_data          = project_path + "data/ds009/"
location_of_images    = project_path + "images/"
location_of_functions = project_path + "code/utils/functions/" 
final_data            = "../../final/data/"
smooth_data           =  final_data + 'smooth/'
hrf_data              = final_data + 'hrf/'
behav_suffix           = "/behav/task001_run001/behavdata.txt"


sys.path.append(location_of_functions)

sub_list = os.listdir(path_to_data)
sub_list = [i for i in sub_list if 'sub' in i]



from event_related_fMRI_functions import hrf_single, np_convolve_30_cuts
from time_shift import time_shift, make_shift_matrix, time_correct
from glm import glm_multiple, glm_diagnostics
from noise_correction import mean_underlying_noise, fourier_predict_underlying_noise,fourier_creation
from hypothesis import t_stat_mult_regression, t_stat
from Image_Visualizing import present_3d, make_mask
from mask_phase_2_dimension_change import masking_reshape_start, masking_reshape_end
from model_comparison import adjR2, BIC, AIC, BIC_2,AIC_2

# Progress bar
sys.stdout.write("Model Selection:   (be slightly patient between subjects)    \n")

#for input_var in ['adjR2','BIC','AIC']:
 
adjr2_1 = []
aic_1 = []
bic_1 = []
adjr2_2 = []
aic_2 = []
bic_2 = []
adjr2_3 = []
aic_3 = []
bic_3 = []
adjr2_4 = []
aic_4 = []
bic_4 = []
adjr2_4_5 = []
aic_4_5 = []
bic_4_5 = []
adjr2_5 = []
aic_5 = []
bic_5 = []
adjr2_6 = []
aic_6 = []
bic_6 = []
adjr2_7 = []
aic_7 = []
bic_7 = []
adjr2_8 = []
aic_8 = []
bic_8 = []
adjr2_9 = []
aic_9 = []
bic_9 = []
adjr2_9_5 = []
aic_9_5 = []
bic_9_5 = []
adjr2_10 = []
aic_10 = []
bic_10 = []


toolbar_width=34

# Load data in
# We compared on subjects 2,3, and 14 as we believed they 
# were represenatative of the entire data set

for i in ['sub002','sub003','sub014']:

    sys.stdout.write(i + ": " + "[%s]" % (" " * toolbar_width))
    sys.stdout.write("\b" * (toolbar_width + 1))

    img = nib.load(smooth_data + i + "_bold_smoothed.nii")
    data = img.get_data() 

    behav = pd.read_table(path_to_data + i + behav_suffix, sep = " ")
    num_TR = float(behav["NumTRs"])
    n_vols = num_TR

    hrf_matrix_all = np.loadtxt("../data/hrf/" + i + "_hrf_all.txt")
    hrf_matrix_1   = np.loadtxt("../data/hrf/" + i + "_hrf_1.txt")
    hrf_matrix_2   = np.loadtxt("../data/hrf/" + i + "_hrf_2.txt")
    hrf_matrix_3   = np.loadtxt("../data/hrf/" + i + "_hrf_3.txt")



    mask = nib.load(path_to_data + i + '/anatomy/inplane001_brain_mask.nii.gz')
    mask_data = mask.get_data()
    mask_data = make_mask(np.ones(data.shape[:-1]), mask_data, fit = True)
    mask_data = mask_data != 0
    mask_data = mask_data.astype(int)

    # PCA

    to_2d = masking_reshape_start(data, mask_data)

    X_pca = to_2d - np.mean(to_2d,0) - np.mean(to_2d,1)[:, None]

    cov = X_pca.T.dot(X_pca)

    U, S, V = npl.svd(cov)
    pca_addition = U[:, :6] # ~40% coverage



    #START DOING GLM
    for j in range(data.shape[2]):

        data_slice = data[:,:,j,:]
        mask_slice = mask_data[:,:,j]
        data_slice = data_slice.reshape((-1, int(num_TR)))
        mask_slice = np.ravel(mask_slice)

        data_slice = data_slice[mask_slice == 1]

        # all conditions in 1 roof (cond_all)
        X = np.ones((n_vols,15))
        X[:, 1] = hrf_matrix_all[:, j] # 1 more
        X[:, 2] = np.linspace(-1, 1, num = X.shape[0]) #drift
        X[:, 3:9] = fourier_creation(X.shape[0], 3)[:, 1:] # six more
        X[:, 9:] = pca_addition

        # all conditions seperate (cond1, cond2, cond3)
        X_cond = np.ones((n_vols,17))
        X_cond[:, 1] = hrf_matrix_1[:, j] # 1 more
        X_cond[:, 2] = hrf_matrix_2[:, j] # 1 more
        X_cond[:, 3] = hrf_matrix_3[:, j] # 1 more
        X_cond[:, 4] = np.linspace(-1, 1, num = X.shape[0]) #drift # one 
        X_cond[:, 5:11] = fourier_creation(X.shape[0], 3)[:, 1:] # six more
        X_cond[:, 11:] = pca_addition


    #START CREATING MODELS

        ###################
        #   MODEL 1       #
        ###################
        # hrf (simple)

        beta1, t,df1, p = t_stat_mult_regression(data_slice, X[:, 0:2])
    
        MRSS1, fitted, residuals = glm_diagnostics(beta1, X[:, 0:2], data_slice)

        model1_slice = np.zeros(len(MRSS1))
    
        rank1 = npl.matrix_rank(X[:, 0:2])
    
        count = 0

        for value in MRSS1:
            model1_slice[count] = adjR2(value, np.array(data_slice[count, :]), df1, rank1)  
            count += 1


        adjr2_1 = adjr2_1 + model1_slice.tolist()

        aic_1 = aic_1 + AIC_2(MRSS1, data_slice, df1, rank1).tolist()
        bic_1 = bic_1 + BIC_2(MRSS1, data_slice, df1, rank1).tolist()

        ###################
        #   MODEL 2       #
        ###################

        # hrf + drift

        beta2, t, df2, p = t_stat_mult_regression(data_slice, X[:, 0:3])
    
        MRSS2, fitted, residuals = glm_diagnostics(beta2, X[:, 0:3], data_slice)

        model2_slice = np.zeros(len(MRSS2))
    
        rank2 = npl.matrix_rank(X[:, 0:3])
        count = 0

        for value in MRSS2:
            model2_slice[count] = adjR2(value, np.array(data_slice[count, :]), df2, rank2)  
            count += 1

        adjr2_2 = adjr2_2 + model2_slice.tolist()

        aic_2 = aic_2 + AIC_2(MRSS2, data_slice, df2, rank2).tolist()
        bic_2 = bic_2 + BIC_2(MRSS2, data_slice, df2, rank2).tolist()

        ###################
        #   MODEL 3       #
        ###################

        # 1.3 hrf + drift + fourier

        beta3, t, df3, p = t_stat_mult_regression(data_slice, X[:, 0:9])
    
        MRSS3, fitted, residuals = glm_diagnostics(beta3, X[:, 0:9], data_slice)

        model3_slice = np.zeros(len(MRSS3))
    
        rank3 = npl.matrix_rank(X[:, 0:9])
        count = 0

        for value in MRSS3:
            model3_slice[count] = adjR2(value, np.array(data_slice[count, :]), df3, rank3)  
            count += 1


        adjr2_3 = adjr2_3 + model3_slice.tolist()

        aic_3 = aic_3 + AIC_2(MRSS3, data_slice, df3, rank3).tolist()
        bic_3 = bic_3 + BIC_2(MRSS3, data_slice, df3, rank3).tolist()

        ###################
        #   MODEL 4       #
        ###################

        # 1.4 hrf + drift + pca 6

        beta4, t, df4, p = t_stat_mult_regression(data_slice, X[:, [0, 1, 2, 7, 10, 11, 12, 13, 14]])
    
        MRSS4, fitted, residuals = glm_diagnostics(beta4, X[:, [0, 1, 2, 7, 10, 11, 12, 13, 14]], data_slice)

        model4_slice = np.zeros(len(MRSS4))
        rank4 = npl.matrix_rank(X[:, [0, 1, 2, 7, 10, 11, 12, 13, 14]])
        count = 0

        for value in MRSS4:
            model4_slice[count] = adjR2(value, np.array(data_slice[count, :]), df4, rank4)  
            count += 1

        adjr2_4 = adjr2_4 + model4_slice.tolist()

        aic_4 = aic_4 + AIC_2(MRSS4, data_slice, df4, rank4).tolist()
        bic_4 = bic_4 + BIC_2(MRSS4, data_slice, df4, rank4).tolist()

        ###################
        #   MODEL 4_5       #
        ###################

        #hrf + drift + pca 4

        beta4_5, t, df4_5, p = t_stat_mult_regression(data_slice, X[:, [0, 1, 2, 7, 10, 11, 12]])
    
        MRSS4_5, fitted, residuals = glm_diagnostics(beta4_5, X[:, [0, 1, 2, 7, 10, 11, 12]], data_slice)
        rank4_5 = npl.matrix_rank(X[:, [0, 1, 2, 7, 10, 11, 12]])

        model4_5_slice = np.zeros(len(MRSS4_5))
        count = 0

        for value in MRSS4_5:
            model4_5_slice[count] = adjR2(value, np.array(data_slice[count, :]), df4_5, rank4_5)  
            count += 1

        adjr2_4_5 = adjr2_4_5 + model4_5_slice.tolist()

        aic_4_5 = aic_4_5 + AIC_2(MRSS4_5, data_slice, df4_5, rank4_5).tolist()
        bic_4_5 = bic_4_5 + BIC_2(MRSS4_5, data_slice, df4_5, rank4_5).tolist()

        ###################
        #   MODEL 5       #
        ###################

        #hrf + drift + pca 6 + fourier

        beta5, t, df5, p = t_stat_mult_regression(data_slice, X)
    
        MRSS5, fitted, residuals = glm_diagnostics(beta5, X, data_slice)

        model5_slice = np.zeros(len(MRSS5))
    
        rank5 = npl.matrix_rank(X)
    
        count = 0

        for value in MRSS5:
            model5_slice[count] = adjR2(value, np.array(data_slice[count, :]), df5, rank5)  
            count += 1

        adjr2_5 = adjr2_5 + model5_slice.tolist()

        aic_5 = aic_5 + AIC_2(MRSS5, data_slice, df5, rank5).tolist()
        bic_5 = bic_5 + BIC_2(MRSS5, data_slice, df5, rank5).tolist()

        ###################
        #   MODEL 6       #
        ###################

        #hrf

        beta6, t, df6, p = t_stat_mult_regression(data_slice, X_cond[:, 0:4])

        MRSS6, fitted, residuals = glm_diagnostics(beta6, X_cond[:, 0:4], data_slice)

        model6_slice = np.zeros(len(MRSS6))
    
        rank6 = npl.matrix_rank(X_cond[:, 0:4])
    
        count = 0

        for value in MRSS6:
            model6_slice[count] = adjR2(value, np.array(data_slice[count, :]), df6, rank6)
            count += 1

        adjr2_6 = adjr2_6 + model6_slice.tolist()

        aic_6 = aic_6 + AIC_2(MRSS6, data_slice, df6, rank6).tolist()
        bic_6 = bic_6 + BIC_2(MRSS6, data_slice, df6, rank6).tolist()

        ###################
        #   MODEL 7       #
        ###################

        #hrf + drift

        beta7,t,df7,p = t_stat_mult_regression(data_slice, X_cond[:, 0:5])

        MRSS7, fitted, residuals = glm_diagnostics(beta7, X_cond[:, 0:5], data_slice)

        model7_slice = np.zeros(len(MRSS7))
    
        rank7 = npl.matrix_rank(X_cond[:, 0:5])
    
        count = 0

        for value in MRSS7:
            model7_slice[count] = adjR2(value, np.array(data_slice[count, :]), df7, rank7)
            count += 1


        adjr2_7 = adjr2_7 + model7_slice.tolist()

        aic_7 = aic_7 + AIC_2(MRSS7, data_slice, df7, rank7).tolist()
        bic_7 = bic_7 + BIC_2(MRSS7, data_slice, df7, rank7).tolist()

        ###################
        #   MODEL 8       #
        ###################

        #hrf + drift + fourier

        beta8, t, df8, p = t_stat_mult_regression(data_slice, X_cond[:, 0:11])

        MRSS8, fitted, residuals = glm_diagnostics(beta8, X_cond[:, 0:11], data_slice)

        model8_slice = np.zeros(len(MRSS8))
    
        rank8 = npl.matrix_rank(X_cond[:, 0:11])
    
        count = 0

        for value in MRSS8:
            model8_slice[count] = adjR2(value, np.array(data_slice[count, :]), df8, rank8)
            count += 1

        adjr2_8 = adjr2_8 + model8_slice.tolist()

        aic_8 = aic_8 + AIC_2(MRSS8, data_slice, df8, rank8).tolist()
        bic_8 = bic_8 + BIC_2(MRSS8, data_slice, df8, rank8).tolist()

        ###################
        #   MODEL 9       #
        ###################

        #hrf + drift + pca 6


        beta9,t,df9,p = t_stat_mult_regression(data_slice, X_cond[:, list(range(5)) + list(range(11, 17))])

        MRSS9, fitted, residuals = glm_diagnostics(beta9, X_cond[:, list(range(5)) + list(range(11, 17))], data_slice)

        model9_slice = np.zeros(len(MRSS9))
    
        rank9 = npl.matrix_rank(X_cond[:, list(range(5)) + list(range(11, 17))])
    
        count = 0

        for value in MRSS9:
            model9_slice[count] = adjR2(value, np.array(data_slice[count, :]), df9, rank9)
            count += 1

        adjr2_9 = adjr2_9 + model9_slice.tolist()

        aic_9 = aic_9 + AIC_2(MRSS9, data_slice, df9, rank9).tolist()
        bic_9 = bic_9 + BIC_2(MRSS9, data_slice, df9, rank9).tolist()

        #####################
        #   MODEL 9_5       #
        #####################

        #hrf + drift + pca 4


        beta9_5, t, df9_5, p = t_stat_mult_regression(data_slice, X_cond[:, list(range(5)) + list(range(11, 15))])

        MRSS9_5, fitted, residuals = glm_diagnostics(beta9_5,X_cond[:, list(range(5)) + list(range(11, 15))], data_slice)

        model9_5_slice = np.zeros(len(MRSS9_5))
    
        rank9_5 = npl.matrix_rank(X_cond[:, list(range(5)) + list(range(11, 15))])
    
        count = 0

        for value in MRSS9_5:
            model9_5_slice[count] = adjR2(value, np.array(data_slice[count, :]), df9_5, rank9_5)
            count += 1


        adjr2_9_5 = adjr2_9_5 + model9_5_slice.tolist()

        aic_9_5 = aic_9_5 + AIC_2(MRSS9_5, data_slice, df9_5, rank9_5).tolist()
        bic_9_5 = bic_9_5 + BIC_2(MRSS9_5, data_slice, df9_5, rank9_5).tolist()
        
        ###################
        #   MODEL 10       #
        ###################

        #hrf + drift + pca 6 + fourier
        
        beta10,t,df10,p = t_stat_mult_regression(data_slice, X_cond)

        MRSS10, fitted, residuals = glm_diagnostics(beta10, X_cond, data_slice)

        model10_slice = np.zeros(len(MRSS10))
    
        rank10 = npl.matrix_rank(X_cond)
    
        count = 0

        for value in MRSS10:
            model10_slice[count] = adjR2(value, np.array(data_slice[count, :]), df10, rank10)
            count += 1

        adjr2_10 = adjr2_10 + model10_slice.tolist()

        aic_10 = aic_10 + AIC_2(MRSS10, data_slice, df10, rank10).tolist()
        bic_10 = bic_10 + BIC_2(MRSS10, data_slice, df10, rank10).tolist()

        sys.stdout.write("-")
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

    

aic_hold = np.array([np.mean(aic_1), np.mean(aic_2), np.mean(aic_3), np.mean(aic_4), np.mean(aic_4_5), np.mean(aic_5), np.mean(aic_6), np.mean(aic_7),
np.mean(aic_8), np.mean(aic_9),np.mean(aic_9_5), np.mean(aic_10)])  

bic_hold = np.array([np.mean(bic_1), np.mean(bic_2), np.mean(bic_3), np.mean(bic_4), np.mean(bic_4_5), np.mean(bic_5), np.mean(bic_6), np.mean(bic_7),
np.mean(bic_8), np.mean(bic_9),np.mean(bic_9_5), np.mean(bic_10)])  

adjr2_hold = np.array([np.mean(adjr2_1), np.mean(adjr2_2), np.mean(adjr2_3), np.mean(adjr2_4), np.mean(adjr2_4_5), np.mean(adjr2_5), np.mean(adjr2_6), np.mean(adjr2_7),
np.mean(adjr2_8), np.mean(adjr2_9),np.mean(adjr2_9_5), np.mean(adjr2_10)])  

aic_hold = aic_hold.reshape((2, 6))
bic_hold = bic_hold.reshape((2, 6))
adjr2_hold = adjr2_hold.reshape((2, 6))

np.savetxt('../data/model_comparison/' + "AIC_2" + '.txt', aic_hold)
np.savetxt('../data/model_comparison/' + "BIC_2" + '.txt', bic_hold)
np.savetxt('../data/model_comparison/' + "AdjR2_2" + '.txt', adjr2_hold)


