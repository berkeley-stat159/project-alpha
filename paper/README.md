## Generating the Report

The Makefile contains recipes to generate the report. 

- `make all`: Generates the PDF of our report, including appendices. 

- `make clean`: `make all` generates a lot of intermediate files besides the desired PDF, so `make clean` will remove all these other files. 
- `make reset`: does the same thing as make clean, but also removes the report PDF, effectively reseting the directory to the state it was in prior to calling `make all`.

Note that the raw `report.tex` file does not contain any content outside of section headers. To view the raw content for each section, please refer to the individual `.tex` files located in the `main_sections` and `appendix` subdirectories. Images required to generate the report are cached in the `project-alpha/images`  directory, but may also be reproduced by navigating to the `project-alpha/final`directory and running `make all`.
