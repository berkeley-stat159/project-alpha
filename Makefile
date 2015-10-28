.PHONY: all clean coverage test

all: clean

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

coverage:
	nosetests code/utils data --with-coverage --cover-package=data  --cover-package=utils

test:
	nosetests code/utils data

verbose:
	nosetests -v code/utils data
