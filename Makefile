.PHONY: all clean coverage test 

all: clean

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

coverage:
	nosetests code/utils/tests --with-coverage --cover-package=code/utils/functions
	nosetests data/tests --with-coverage --cover-package=data/get_hashes.py --cover-package=data/get_all_hashes.py

test:
	nosetests code/utils/tests/ 
	nosetests data/tests 

verbose:
	nosetests code/utils/tests/ -v
	nosetests data/tests -v
