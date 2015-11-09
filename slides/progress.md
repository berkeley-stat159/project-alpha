% Project Alpha Progress Report
% Kent Chen, Rachel Lee, Ben LeRoy, Jane Liang, Hiro Udagawa
% November 12, 2015

# Background

## The Paper

- from OpenFMRI.org
- "The Generality of Self-Control" (Jessica Cohen, Russell Poldrack)

## The Data

- 24 subjects
- 3 conditions per subject

## Methods/ Preliminary Results

# Convolution
- Worked with problems with event-related stimulus model

# Smoothing
- Convolution with a Gaussian filter (scipy module)

# Linear regression
- Multiple and single regression with stimulus (all conditions and seperate)

# Hypothesis testing
- General t-tests on $\beta$ values
- Across suject analysis

# Time series
- ARIMA(1,1,1) model

# PCA
- Modeling against
- SVD

# Discussion
- Data processing and methods for modeling voxel time courses
- Linear regression, hypothesis models
- Principal components analysis and modeling individual volumes as a time series

# Future Goals and Work
- Explore modeling more of the noise in our data
- Realignment of scans to correct for the time it takes to scan each voxel compared to the start of the scan
- More quantitative and robust indicators for validating time series models should be implemented