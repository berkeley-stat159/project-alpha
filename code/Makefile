.PHONY: all figs misc clean test coverage 

all:
	cd utils/scripts && python bh_script.py
	cd utils/scripts && python cluster.py
	cd utils/scripts && python convolution_appendix_plots.py
	cd utils/scripts && python event_related_HRF_script.py
	cd utils/scripts && python get_pcs_script.py
	cd utils/scripts && python glm_script.py
	cd utils/scripts && python hypothesis_script.py
	cd utils/scripts && python lin_reg_plots.py
	cd utils/scripts && python mean_across.py
	cd utils/scripts && python model_selection.py
	cd utils/scripts && python multi_regression_script.py
	cd utils/scripts && python noise_correction_script.py
	cd utils/scripts && python normality_script.py
	cd utils/scripts && python outliers_script.py
	cd utils/scripts && python pca_script.py
	cd utils/scripts && python smooth_script.py
	cd utils/scripts && python tgrouping_script.py
	cd utils/scripts && python time_shift_script.py
	cd utils/scripts && python tsa_script.py

# NOT COMPLETED: lin_reg_plots? 
figs: 
	cd utils/scripts && python tsa_script.py

misc:
	cd utils/misc && ipython nbconvert BART_Data_Beginning.ipynb 

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

test:
	nosetests code/utils/tests -w ..

coverage:
	nosetests code/utils/tests -w .. --with-coverage --cover-package=utils/functions
