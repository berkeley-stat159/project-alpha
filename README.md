# Project Alpha

Statistics 159/259: Reproducible and Collaborative Statistical Data Science

UC Berkeley

Fall 2015 

[![Build Status](https://travis-ci.org/berkeley-stat159/project-alpha.svg?branch=master)](https://travis-ci.org/berkeley-stat159/project-alpha?branch=master)
[![Coverage Status](https://coveralls.io/repos/berkeley-stat159/project-alpha/badge.svg?branch=master)](https://coveralls.io/r/berkeley-stat159/project-alpha?branch=master)

This repository stores the documentation of our analysis of the balloon-analogue risk task (BART) data included as part of the OpenfMRI ds000009 data set [*The generality of self-control*](https://openfmri.org/dataset/ds000009/). The original analysis was conducted by Jessica Cohen for her doctoral thesis, under the advisement of Russell Poldrack and was largely focused on comparing different notions of self-control across several fMRI studies. Our aim is to reproduce the original analysis and identify regions of the brain with high activation levels over the course of the BART study. 

Many thanks to Jarrod Millman, Matthew Brett, J-B Poline, and Ross Barnowski for their advice and encouragement. 


## Navigating the Repository 

The Makefile contains four commands: `clean`, `test`, `verbose`, and `coverage`. 
- `make clean`: Remove all extra files generated when compiling code. Does this recursively for all subdirectories. 
- `make test`: Tests the functions located in the `data` and `code` directories, to be used to validate and analyze the data, respectively. 
- `make verbose`: Performs the same actions as `make test`, but uses the verbose nosetests option. 
- `make coverage`: Generates a coverage report for the functions located in the `data` and `code` directories. 

NEED TO ADD: Info on navigating the subdirectories. Images?
Instructions on what to do to get the analysis. 

## Contributers 

- Kent Chen ([`kentschen`](https://github.com/kentschen))
- Rachel Lee ([`reychil`](https://github.com/reychil))
- Benjamin LeRoy ([`benjaminleroy`](https://github.com/benjaminleroy))
- Jane Liang ([`janewliang`](https://github.com/janewliang))
- Hiroto Udagawa ([`hiroto-udagawa`](https://github.com/hiroto-udagawa))
