## Downloading and Validating the Data

The Makefile contains six commands: `data`, `testing_data`, `validate`, `testing_validate`, `test`, and `coverage`. 
- `make data`: Downloads and decompresses the .tgz archive of the data for our analysis from OpenfMRI.org. We are using the BART data from ds009, *The generality of self-control*. All files will be located in the `ds009` subdirectory. A complete download will take a while (~ 6 GB). 
- `make testing_data`: Downloads three data files from the ds114 OpenfMRI data set, hosted at jarrodmillman.com. These are the data files that we worked on in class, and will be used in this project strictly for testing functions. All files will be located in the `ds114` subdirectory. 

- `make validate`: Validates the ds009 data after downloading it via `make data`. Checks if all files are present and have the correct hash, based on the hashes stored in the `ds009_hashes.json file`. 
- `make testing_validate`: Validates the ds114 data to be used strictly for testing after downloading it via `make testing_data`. Checks if all three file are present and have the correct hash. 

- `make test`: Tests the functions associated with validating data (naming obtaining and checking file hashes). 
- `make coverage`: Generates a coverage report for the functions associated with validating data. 
