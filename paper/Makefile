TITLE="report"
APP_CONV = "appendix_convolution"
APP_BH = "appendix_bh"
APP_CLUSTER = "appendix_clustering"
APP_TIME = "appendix_time"

all:
	pdflatex $(TITLE).tex
	bibtex $(TITLE)
	pdflatex $(TITLE).tex
	pdflatex $(TITLE).tex
	
	pdflatex appendix/$(APP_CONV).tex
	#bibtex $(APP_CONV)
	#pdflatex appendix/$(APP_CONV).tex
	#pdflatex appendix/$(APP_CONV).tex
	
	pdflatex appendix/$(APP_BH).tex
	#bibtex $(APP_BH)
	#pdflatex appendix/$(APP_BH).tex
	#pdflatex appendix/$(APP_BH).tex
	
	pdflatex appendix/$(APP_CLUSTER).tex
	#bibtex $(APP_CLUSTER)
	#pdflatex appendix/$(APP_CLUSTER).tex
	#pdflatex appendix/$(APP_CLUSTER).tex
	
	pdflatex appendix/$(APP_TIME).tex
	bibtex $(APP_TIME)
	pdflatex appendix/$(APP_TIME).tex
	pdflatex appendix/$(APP_TIME).tex
	
clean:
	rm -f *.{aux,log,bbl,lof,lot,blg,out}
