.PHONY: all clean coverage test 

all: clean

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

coverage:
	nosetests code/utils/tests data/tests --with-coverage --cover-package=code/utils/functions,data/get_hashes.py,data/get_all_hashes.py

test:
	nosetests code/utils/tests/ data/tests 

verbose:
	nosetests code/utils/tests/ data/tests -v
