.PHONY: data images all clean  

data:
	cd data_scripts && python smooth_final.py
	cd data_scripts && python convolution_final.py
	cd data_scripts && python selection_final.py
	cd data_scripts && python glm_final.py
	cd data_scripts && python hypothesis_final.py
	cd data_scripts && python bh_t_beta_final.py

images: 
	cd image_scripts && python ben_final.py 
	cd image_scripts && python bh_t_beta_plots.py 
	cd image_scripts && python convolution_appendix_plots.py 
	cd image_scripts && python event_related_plots.py
	cd image_scripts && python image_overlay.py 
	cd image_scripts && python lin_reg_plots.py 
	cd image_scripts && python normality_final.py 
	cd image_scripts && python parameter_selection_plots.py 
	cd image_scripts && python pca_plots.py 
	cd image_scripts && python selection_plots.py
	cd image_scripts && python tsa_script.py
	cd image_scripts && python ward_plots.py

all: 
	make data
	make images

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f
