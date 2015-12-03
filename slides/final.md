% Progress Report for Project Alpha
% Kent Chen, Rachel Lee, Ben LeRoy, Jane Liang, Hiroto Udagawa
% December 1, 2015



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




# Convolution: Worked with problems with event-related stimulus model

\begin{figure}[ht]
\centering
	\begin{minipage}[b]{0.5\linewidth}
		\centering
		\includegraphics[width=1\linewidth]{images/convolution_vs_neural_stimulus}  
		% needs to be from the event_related_HRF_script2.py 
		\caption{\scriptsize{Different convolution functions vs. the Neural stimulus}}
		\label{fig:convolution}

	\end{minipage}
\quad
	\begin{minipage}[b]{0.45\linewidth}
		\centering
		\scriptsize{\begin{tabular}{|l | c|}
		\hline
		name in graph & Speed per loop \\
		\hline
		np    			 & 14.4 µs \\
		user 2     		 & 972 ms  \\
		user 3     		 & 1.15 s  \\
		user 4 (15 cuts) & 98.3 ms \\
		user 4 (30 cuts) & 185 ms  \\
		user 5     	 	 & 110 ms  \\
		\hline
		\end{tabular}}
		\vspace{5mm}
		\caption{\scriptsize{Speed to create HRF predictions for Subject 001, all conditions}}
		\label{table:convolution}
	\end{minipage}
\end{figure}

# Smoothing: Convolution with a Gaussian filter (scipy module)

\begin{figure}
  \centering
  {\includegraphics[scale=0.25]{images/original_slice.png}}{\includegraphics[scale=0.25]{images/smoothed_slice.png}}
\end{figure}


# Linear regression: Single and multiple regression with stimulus (all conditions and seperate)

\begin{figure}[ht]
\centering
\begin{minipage}[b]{0.45\linewidth}
	\centering
	\includegraphics[width=.8\linewidth]{images/Fitted_v_Actual.png} 
	\caption{Fitted vs Actual}
	\label{fig:fit_vs_act}
\end{minipage}	
\quad
\begin{minipage}[b]{0.45\linewidth}
	\centering
		\includegraphics[width=.8\linewidth]{images/Fitted_v_Residuals.png} 
	\caption{Fitted vs Residual}
	\label{fig:fit_vs_res}
\end{minipage}
\end{figure}



#  Hypothesis testing: General t-tests on $\beta$ values, and across subject analysis


\begin{figure}[ht]
\centering
\begin{minipage}[b]{0.45\linewidth}
	\centering
	\includegraphics[width=.8\linewidth]{images/hypothesis_testing.png} 
	\caption{Smoothed t-values}
	\label{fig:t-value}
\end{minipage}	
\quad
\begin{minipage}[b]{0.45\linewidth}
	\centering
		\includegraphics[width=.8\linewidth]{images/hypothesis_testing2.png} 
	\caption{unsmoothed t-values}
	\label{fig:t-value2}
\end{minipage}
\end{figure}

# PCA on the voxel by time covariance matrix.


\begin{figure}
  \centering
  {\includegraphics[scale=0.5]{images/pcasub010.png}}
\end{figure}


# Clustering

\begin{figure}[ht]
\centering
\begin{minipage}[b]{0.6\linewidth}
	\centering
	\includegraphics[width=.8\linewidth]{images/clustering1.png} 
	\caption{Clustering 1}

\end{minipage}	
\quad
\hspace{-30mm}
\begin{minipage}[b]{0.6\linewidth}
	\centering
	\includegraphics[width=.8\linewidth]{images/clustering2.png} 
	\caption{Clustering 2}

\end{minipage}
\end{figure}



