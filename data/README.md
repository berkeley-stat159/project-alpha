## Downloading and Validating the Data

The Makefile contains recipes to download and validate the data required for 
our analysis and tests. Note that downloading the ds114 testing data is not 
required to perform our analysis, but is needed to run the tests for our 
functions. 

- `make data`: Downloads and decompresses the `.tgz` archive of the data for 
our analysis from OpenfMRI.org. We are using the BART data from ds009, 
*The generality of self-control*. All downloaded files will be located in the 
`ds009` subdirectory. Note that the `.tgz` archive is about 6 GB, so users 
should plan accordingly for space and download time. 
- `make testing_data`: Downloads three data files from the ds114 OpenfMRI data 
set, hosted at jarrodmillman.com. These are the data files that we worked on 
in class, and will be used in this project strictly for testing functions. All 
downloaded files will be located in the `ds114` subdirectory. The three data 
files together use 42.5 MB of space, so download time should be reasonable.

- `make validate`: Validates the ds009 data after downloading it via 
`make data`. Checks if all files are present and have the correct hash, based 
on the hashes stored in the `ds009_hashes.json file`. 
- `make testing_validate`: Validates the ds114 data to be used strictly for 
testing after downloading it via `make testing_data`. Checks if all three file 
are present and have the correct hash. 

- `make test`: Tests the functions associated with validating data (naming 
obtaining and checking file hashes). 
- `make coverage`: Generates a coverage report for the functions associated 
with validating data. 
