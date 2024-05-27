# The raw data data (e.g. model grids) associated to this project are available on Zenodo. 
## Due to the volume of data, the tracks are spread across four Zenodo DOIs and several different .tar.gz files

### https://zenodo.org/records/11280179
### https://zenodo.org/records/11282597
### https://zenodo.org/records/11353933
### https://zenodo.org/records/11357395

### All phase 1 and phase 2 evolutionary tracks are contained in the files 
#### ZAMS_to_TCHeB_y-fixed.tar.gz
#### TCHeB_to_AGB_y-fixed.tar.gz
#### ZAMS_to_TCHeB_y-varied.tar.gz
#### TCHeB_to_AGB_y-varied.tar.gz
### sorted by whether a varied or static helium assumption was used. 


### To use the smalled number of separate Zenodo DOIs possible, not all data grouped in the same way. The tar.gz files labeled with a string of the form "AGB_to_end_M1.XX_y-fixed.tar.gz" include all models having a mass beginning with 1 (1.00, 1.10, 1.20, etc) and adopting the fixed helium assumption. Files labeled in the form "AGB_to_end_M1.XX_y-varied.tar.gz" include the same, but using the helium-varied assumption. 

###  Files labeled with a string of the form "AGB_to_end_M3.10_all.tar.gz" include both the fixed-helium and varied-helium tracks for all masses that start with 3 (3.00, 3.10, 3.20, etc).

<!-- ### helium-fixed grid
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



 -->
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