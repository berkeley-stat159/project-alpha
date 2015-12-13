## Analyzing the Data

The Makefile contains recipes to perform the intermediate analysis for the 
project, as well as generate some supplementary work and test all user-defined 
functions. The results of this directory should largely be regarded as 
transitional (but nonetheless valuable) steps in the progression of our work. 
Many of the scripts run processes that are no longer in use, or primarily 
serve as illustrative examples on how to utilize certain functions for a 
single subject. For those strictly interested in our final analysis, we 
recommend looking at our `project-alpha/final` directory instead. 

- `make all`: Performs all intermediate data analysis. Does not include the 
supplementary work rendered by `make misc`. 
- `make figs`: Runs only the scripts needed to reproduce figures that are used 
in the report and slides. Note that these figures is not exhaustive: please 
navigate to `project-alpha/final` and run `make all` there to reproduce all 
figures required for the report and slides. 
- `make misc`: Generates supplementary work. More specifically, it renders an 
iPython Notebook with early exploratory data analysis into a viewable HTML 
file in the `utils/misc` subdirectory. 

- `make clean`: Removes all extra files generated when compiling code. Does 
this recursively for all subdirectories. 
- `make test`: Tests all user-defined functions associated with analyzing the 
data. 
- `make coverage`: Generates a coverage report for all user-defined functions 
associated with analyzing the data. 

Additional documentation and information on the subdirectories of `utils` can 
be found in their respective READMEs. 

- `utils/functions`: Code files for all user-defined functions used for 
analysis. 
- `utils/tests`: Test files for the user-defined functions. 
- `utils/scripts`: Scripts to run the intermediate analysis. 
- `utils/misc`: Supplementary analysis and documentation. 

All output figures are saved in `project-alpha/images`, which also caches the 
figures required for the report and slides. These figures can be reproduced in 
their entirety by navigating to `project-alpha/final` and running `make all`. 
The `make figs` recipe in this directory only generates the figures used in 
the report and slides that are output by scripts in this directory. 
