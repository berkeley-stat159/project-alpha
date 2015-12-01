## How to generate our report. 

The Makefile contains four commands: `all`, `app`, `clean`, and `reset`. 
- `make all`: Gennerates the PDF of our main report. 
- `make app`: Generates the additional PDFs containing our appendices. 
- `make clean`: `make all` and `make app` generate a lot of intermediate files besides the desired PDFs, so `make clean` will remove all these other files. 
- `make reset`: does the same thing as make clean, but also removes the PDFs, effectively reseting the directory to the state it was in prior to calling `make all` and/or `make app`.
