# The raw data data (e.g. model grids) associated to this project are available on Zenodo. 
## Due to the volume of data, the tracks are spread across multiple Zenodo DOIs and 14 different .tar.gz files

### helium-fixed grid
#### phase 1
ZAMS_to_TCHeB_y-fixed.tar.gz

#### phase 2
TCHeB_to_AGB_y-fixed.tar.gz

#### phase 3
AGB_to_end_M1.XX_y-fixed.tar.gz  
AGB_to_end_M2.XX_y-fixed.tar.gz 
AGB_to_end_M3.XX_y-fixed.tar.gz  
AGB_to_end_M4.XX_y-fixed.tar.gz
AGB_to_end_M5.XX_y-fixed.tar.gz

### helium-varied grid
#### phase 1
ZAMS_to_TCHeB_y-varied.tar.gz

#### phase 2
TCHeB_to_AGB_y-varied.tar.gz

#### phase 3
AGB_to_end_M1.XX_y-varied.tar.gz  
AGB_to_end_M2.XX_y-varied.tar.gz 
AGB_to_end_M3.XX_y-varied.tar.gz  
AGB_to_end_M4.XX_y-varied.tar.gz
AGB_to_end_M5.XX_y-varied.tar.gz




# AGB_grid_visualizer
visualization tools for grids of AGB models accompanying RHya paper

## includes:
#### - `bokeh_heatmaps.py`
which loads the visualization tool in the browser 
#### - four "best_fits" data files
 containing the best fit statistics as a function of mass, metallicity, and mode ID assumption (FM vs O1), computed according to different assumptions regarding agreement with period, luminosity, effective temperature, and radius. 
	
 The files with "STRICT" in the prefix correspond to an assumption of hard limits on L, Teff, and R in the initial agreement domain, also referred to as a "top-hat" prior on the classical observations. The maps available for each of these files show the weigthed root-mean-square error (w-rmse) statistc based on agreement between the theoretical and observed period measurements only. 
	
 The files without "STRICT" in the prefix correspond to arbitrarily loose assumptions on the initial agreement between theoretical and obserational Teff, L and R. There are two maps available for each of these files: one showing the w-rmse statistic based on agreement between the theoretical and observed period measurements only (as in the STRICT case), and one showing a w-rmse statistic that considers agreement with period as well as with Teff, L, and R, weighted according to the relative observational uncertainty in each parameter.

See paper for more detail on the statistics.  
#### - models.dat
 a list of all models run to at least the onset of the thermally pulsing AGB phase, needed to generate certain aspects of the visualizer. 
 The file names correspond verbatim to the MESA output data files that were used to make the pulse spectra figures in the visualizer and are available at https://meridithjoyce.com/pulse_data/

#### - names_of_pngs.dat
 a list of png files needed in the visualizer. The filenames correspond verbatim to the files used in the visualizer. The files are availble at https://meridithjoyce.com/images/AGB_grid/Nov28_2023/ and can be reproduced by running the `associated_plots.py` script in a directory containing the desired history data files 

#### - model_ages.dat
 crude average age in Myr taken by summing the model star's age at the onset of the TP-AGB phase and its age at the termination of the run and dividing by two.  


#### - `associated_plots.py`
  script used to generate the pngs 

#### - r_hya_supplemented_years.csv
  file containing the composite observational measurements of period and period uncertainty as a function of (relative) time


### The visualizer is supposed to behave like this:
![alt text](https://github.com/mjoyceGR/AGB_grid_visualizer/blob/main/visualizer_screenshot.png?raw=true)