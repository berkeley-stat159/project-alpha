## How to generate our report. 

The Makefile contains three commands: `all`, `clean`, and `reset`. 
- `make all`: Generates the PDF of our report, including appendices. 
- `make clean`: `make all` generate sa lot of intermediate files besides the desired PDF, so `make clean` will remove all these other files. 
- `make reset`: does the same thing as make clean, but also removes the report PDF, effectively reseting the directory to the state it was in prior to calling `make all`.
