## How to insert and reference images correctly. 

- When inserting a figure, make sure it is included in the "images" repo and that you rescale the image appropriately in the report. 
- All figures should have a label and a caption.
- All figure numbers in the text should be called by figure reference. 
- Put BRACKETS around your figure reference in the text.
- Be sure to put the label AFTER the caption (otherwise it won't reference correctly).

Example from time_results.tex:

A comparison between the true observations and the forecasted predictions is shown in [Figure \ref{fig:ts-preds}]. While the forecasted observations look reasonable for approximating the true values, more quantitative metrics for assessing performance need to be implemented. 

\begin{figure}[ht]

\centering

\includegraphics[scale=0.5]{images/ts-preds.png}

\caption{Forecasting the second half of observations based on the first half.}

\label{fig:ts-preds}

\end{figure}
