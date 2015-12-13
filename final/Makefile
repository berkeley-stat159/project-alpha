.PHONY: smooth convolve glm normality conclusion analysis supp figs all clean 

# FIX INDIVIDUAL COMMANDS
smooth:
	cd scripts && python smoothing.py

convolve:
	cd scripts && python convolution.py
	
glm:
	cd scripts && python glm.py

normality:
	cd scripts && normality.python py

conclusion:
	cd scripts && python conclusion.py

# FIX
analysis:
	make smooth
	make convolve
	make glm
	make normality
	make conclusion

# FIX
supp: 

# FIX
figs: 
	cd scripts && python FILENAME
	cd ../code && make figs

all: 
	make analysis
	make eda
	make figs

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f
