## Scripts to Perform Data-Generating Final Analysis

This repository stores the script files for all final data analysis and final 
supplementary data analysis that save data to the `project-alpha/final/data` 
directory (listed in the order that they should be run). 

- `smooth_final.py`: Spatially smooths voxels. 
- `convolution_final.py`: Creates convolved and time-shifted HRFs for each 
subject's voxel time courses. 
- `selection_final.py`: Selects predictors for linear models based on AIC, 
BIC, and adjusted R-squared. 
- `glm_final.py`: Runs linear regression models with linear drift, Fourier, 
and PCA terms. 
- `hypothesis_final.py`: Performs t-tests on the coefficients from linear 
regression. 
- `bh_t_beta_final.py`: Analysis of Benjamini-Hochberg corrected "p-values" 
and t-statistics and beta coefficient estimates above and below certain 
thresholds.



