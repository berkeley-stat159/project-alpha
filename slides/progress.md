% Progress Report for Project Alpha
% Kent Chen, Rachel Lee, Ben LeRoy, Jane Liang, Hiroto Udagawa
% November 12, 2015



# Background

## The Paper

- From OpenFMRI.org (ds009)
- "The Generality of Self-Control" (Jessica Cohen, Russell Poldrack)
<comment about software packages and replication>

## The Data

- BART study with event-related neurological stimulus (balloon demo)
- 24 subjects, 3 conditions per subject
	- Condition 1: Inflation
	- Condition 2: Pop Pop
	- Condition 3: Cash out dem monies
- Download, decompress and check hashes of data


# Initial analysis

- Convolution: Worked with problems with event-related stimulus model

- Smoothing: Convolution with a Gaussian filter (scipy module)

- Linear regression: Single and multiple regression with stimulus (all conditions and seperate)

- Hypothesis testing: General t-tests on $\beta$ values, and across suject analysis

- Time series: ARIMA model

- PCA: Modeling against SVD

# Our Plan

- Hypothesis testing: General t-tests on $\beta$ values, and across suject analysis

- Time series: ARIMA model

- PCA: Modeling against SVD

# Before and After Smoothing
\begin{figure}
  \centering
  {\includegraphics[scale=0.25]{images/original_slice.png}}{\includegraphics[scale=0.25]{images/smoothed_slice.png}}
\end{figure}



# Hypothesis Testing Across Subjects
\begin{figure}
  \centering
  {\includegraphics[scale=0.5]{images/hypothesis_testing.png}}
\end{figure}

# Our Plan

## Goal
- Trying to reproduce methods, but it won't all be the same

## Issues we have encountered
- Convolution with event-related stimuli
- Approach to multiple subjects
- Scan time problems (large dimensions)
- Validation of performance
- Trying to replicate black box analysis

# Our Plan
## What we need to accomplish
- Preprocessing: 
	- Resampling to correct for when the voxels were actual scanned (time shift)
	- Explore Convolution (3rd time's the charm)
- Analysis:
	- Multiple comparision:
		- Permutation test
		- Random field technique
		- Benjamini-Hoffberg

# Comments about our Project

## Most difficult aspect of project?
- Direction of project

## Success in overcoming these obstacles?
- |--------------------------------|
- 		 ^ This successful ^
 
## Most useful parts of class?
- Git workflow

# The Project continued

## What do we need to successfully complete the project?
- Define better goals, and set an end goal
- Take advantage of pre-existing toolkit
- Tie our analysis and conclusions back to the original paper

## Difficulty of making work reproducible?
- Writing tests that maintain our coverage
- Relative paths and making sure nosetests/Makefiles work properly

# Potential topics to cover in class in the future
- Coding best practices and style
- Python approach to machine learning (scikit-learn)
- Other popular software tools used in collaboration
- Learning basics of Pandas


