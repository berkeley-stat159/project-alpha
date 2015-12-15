.PHONY: all misc clean test coverage 

all:
	cd utils/scripts && python bh_script.py > eda.txt
	cd utils/scripts && python cluster.py >> eda.txt
	cd utils/scripts && python event_related_HRF_script.py >> eda.txt
	cd utils/scripts && python get_pcs_script.py >> eda.txt
	cd utils/scripts && python glm_script.py >> eda.txt
	cd utils/scripts && python hypothesis_script.py >> eda.txt
	cd utils/scripts && python mean_across.py >> eda.txt
	cd utils/scripts && python model_selection.py >> eda.txt
	cd utils/scripts && python multiple_ways_imaging_script.py >> eda.txt
	cd utils/scripts && python multi_regression_script.py >> eda.txt
	cd utils/scripts && python noise_correction_script.py >> eda.txt
	cd utils/scripts && python normality_script.py >> eda.txt
	cd utils/scripts && python outliers_script.py >> eda.txt
	cd utils/scripts && python pca_script.py >> eda.txt
	cd utils/scripts && python smooth_script.py >> eda.txt
	cd utils/scripts && python tgrouping_script.py >> eda.txt
	cd utils/scripts && python time_shift_script.py >> eda.txt

misc:
	cd utils/misc && ipython nbconvert BART_Data_Beginning.ipynb 

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

test:
	nosetests code/utils/tests -w ..

coverage:
	nosetests code/utils/tests -w .. --with-coverage --cover-package=utils/functions
