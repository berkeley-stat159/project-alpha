data:
	wget http://openfmri.s3.amazonaws.com/tarballs/ds009_raw.tgz
	tar -zxvf ds009_raw.tgz

testing_data:
	wget -P ds114/ http://www.jarrodmillman.com/rcsds/_downloads/ds114_sub009_t2r1.nii
	wget -P ds114/ http://www.jarrodmillman.com/rcsds/_downloads/ds114_sub009_t2r1_cond.txt
	wget -P ds114/ http://www.jarrodmillman.com/rcsds/_downloads/ds114_sub009_t2r1_conv.txt
	wget http://www.jarrodmillman.com/rcsds/_downloads/ds107_sub001_highres.nii

validate:
	python data.py

testing_validate: 
	python testing_data.py

test:
	nosetests tests -w ..

coverage:
	nosetests tests -w .. --with-coverage --cover-package=get_hashes.py --cover-package=get_all_hashes.py 
