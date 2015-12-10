TITLE="report"

.PHONY: all clean reset

all:
	pdflatex $(TITLE).tex
	bibtex $(TITLE)
	pdflatex $(TITLE).tex
	pdflatex $(TITLE).tex
	
clean:
	find . -name "*.aux" -o -name "*.log" -o -name "*.bbl" -o -name "*.lof" -o -name "*.lot" -o -name "*.blg" -o -name "*.out" | xargs rm -f

reset: 
	find . -name "*.aux" -o -name "*.log" -o -name "*.bbl" -o -name "*.lof" -o -name "*.lot" -o -name "*.blg" -o -name "*.out" -o -name "*.pdf"  | xargs rm -f
