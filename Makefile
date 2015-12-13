.PHONY: all clean test verbose coverage 

all:
	cd data && make data 
	cd data && make validate 
	cd final && make all 

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

test:
	nosetests code/utils/tests/ data/tests 

verbose:
	nosetests code/utils/tests/ data/tests -v

coverage:
	nosetests code/utils/tests data/tests --with-coverage --cover-package=code/utils/functions,data/get_hashes.py,data/get_all_hashes.py
