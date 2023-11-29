# AGB_grid_visualizer
visualization tools for grids of AGB models accompanying RHya paper

## includes:
#### - bokeh_heatmaps.py 
which loads the visualization tool in the browser 
#### - four "best_fits" data files
	 containing the best fit statistics as a function of mass, metallicity, and mode ID assumption (FM vs O1), computed according to different assumptions regarding agreement with period, luminosity, effective temperature, and radius. 
	
	 The files with "STRICT" in the prefix correspond to an assumption of hard limits on L, T, and R in the initial agreement domain, also referred to as a "top-hat" prior on the classical observations. The maps available for each of these files show the weigthed root-mean-square error (w-rmse) statistc based on agreement between the theoretical and observed period measurements only. 
	
	 The files without "STRICT" in the prefix correspond to arbitrarily loose assumptions on the initial agreement between theoretical and obserational Teff, L and R. There are two maps available for each of these files: one showing the w-rmse statistic based on agreement between the theoretical and observed period measurements only (as in the STRICT case), and one showing a w-rmse statistic that considers agreement with period as well as with Teff, L, and R, weighted according to the relative observational uncertainty in each parameter.

	See paper for more detail on the statistics.  
#### - models.dat
	a list of all models run to at least the onset of the thermally pulsing AGB phase, needed to generate certain aspects of the visualizer 
	The file names correspond verbatim to the files plotted to make the pulse spectra figures in the visualizer and are available at https://meridithjoyce.com/pulse_data/

#### - names_of_pngs.dat
	a list of png files needed in the visualizer. The filenames correspond verbatim to the files used in the visualizer and are availble at https://meridithjoyce.com/images/AGB_grid/Nov28_2023/

<!-- #Period    mass   z  pulse_number  period_WRMSE    global_WRMSE 
FM   1.00   0.0013   3   115.0333   283.6358 -->