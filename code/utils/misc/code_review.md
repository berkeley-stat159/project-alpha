## Code Review (Kent):

##General Comments
-Consistency? Such as listing "subject001" instead of subject 1
-Descriptions on the top of scripts, similar to what the smooth_final.py has so far



##Scripts
#glm_script.py

#hypothesis_script.py

#mean_across.py

#multi_regression_script.py

#time_shift_script.py

#model_selection.py

#understanding_convolution.py





## Final Scripts
#conclusion_final.py
-Add in a description on top
 "Final script for our final script"
-Are the commented codes necessary?
 (Lines 70-113)
 -t-stat needs to be added for script to run properly

#convolution_final.py
-Add in a description on top
 "Final script for convolution..."
-Are the commented codes necessary?
 (Lines 1-2): #load condition
              #cond_all
-

#glm_final.py
-Add in a description on top
 "Final script for glm..." 
-Are the commented codes necessary? 
 (Line 47): #img = nib.load(path_to_data+ name+ "/BOLD/task001_run001/bold.nii.gz")
 (Line 55): #data = data[...,num_TR_cut:]
 (Line 61): #residual_final = np.zeros((data.shape))
 (Line 63): #p_final = np.zeros((data.shape[:-1]))
 (Line 80): #p = p[1,:]
 (Line 87-90): #p_final[:,:,j] = p.reshape(data_slice.shape[:-1])
               #t_final2[:,:,j] = t2.reshape(data_slice.shape[:-1])
               #residual_final[:,:,j,:] = residuals.reshape(data_slice.shape)
 (Line 93-95): #np.save("../data/glm/t_stat/"+i+"_tstat2.npy", t_final2)
               #np.save("../data/glm/residual/"+i+"_residual.npy", residual_final)
               #np.save("../data/glm/p-values/"+i+"_pvalue.npy", p_final)

#normality_final.py
-Add in a description on top
 "Final script for glm..."
-Are the commented codes necessary? 
 (Line 8): "subject 1" change to "subject001"
 (Line 22): #sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/")) 
 Should this be renamed to a generalized name or leave as is?
 
#smooth_final.py
-
-Are the commented codes necessary
 Lines(35-40): #Load events2neural from the stimuli module.
               #from stimuli import events2neural
               #from event_related_fMRI_functions import hrf_single, convolution_specialized#
			   #Load our GLM functions. 
			   #from glm import glm, glm_diagnostics, glm_multiple