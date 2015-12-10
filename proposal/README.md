## Generating the Proposal

The Makefile contains three commands: `all`, `clean`, and `reset`. 
- `make all`: Generates the PDF of our proposal. 
- `make clean`: `make all` generates a lot of intermediate files besides the desired PDF, so `make clean` will remove all these other files. 
- `make reset`: does the same thing as make clean, but also removes the proposal PDF, effectively reseting the directory to the state it was in prior to calling `make all`.
