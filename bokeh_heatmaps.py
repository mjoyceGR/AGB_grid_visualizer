#!/usr/bin/env python3
####################################################
#
# Author: M Joyce
#
# see
# https://discourse.bokeh.org/t/adding-a-html-widget-with-images-and-updating-it-similar-to-using-the-hover-tool/2309/3
#
####################################################
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import glob
import sys
import subprocess
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from datetime import datetime 

# sys.path.append('../')
import math_functions_lib as mfl

# sys.path.append('../../py_mesa_reader/')
# import mesa_reader as mr

import pandas as pd
import bokeh
from bokeh.events import ButtonClick
from bokeh.models import ColumnDataSource,  OpenURL, TapTool
from bokeh.plotting import figure, show, output_file 
from bokeh.models import HoverTool
from bokeh.io import show
from bokeh.layouts import row, column
from bokeh.models import Div, HoverTool, CustomJS, ColumnDataSource, Button

import argparse

parser = argparse.ArgumentParser(description='specify fit file and statistic to view')
parser.add_argument('fit_file_name', help='specify best fits file (e.g. "all_best_fits_FM_withmass.dat"\n'+\
									'                             "all_best_fits_O1_withmass.dat"\n'+\
									'                             "STRICT-all_best_fits_FM.dat"\n'+\
									'                             "STRICT-all_best_fits_O1.dat"'									
									, type=str)
parser.add_argument('use_mixed_statistic', help='If n, use period only (recommended). '+\
	                                            'If y, use P+LTR (not available with STRICT domain files). ', type=str)
args = parser.parse_args()
cmdLine=True

#######################################

#fit_files = ['all_best_fits_O1_withmass.dat','all_best_fits_FM_withmass.dat','STRICT-all_best_fits_O1.dat','STRICT-all_best_fits_FM.dat']
fit_file = args.fit_file_name  #all_best_fits_O1_withmass.dat'

if 'STRICT' in fit_file:
	use_mixed_statistic = False
else:
	if args.use_mixed_statistic == 'y':
		use_mixed_statistic = True
	else:
		use_mixed_statistic = False

which_P = fit_file.split('best_fits_')[1].split('_withmass.dat')[0]


if 'STRICT' in fit_file and 'O1' in fit_file:
	mode_domain = 'O1_STRICT'
elif 'STRICT' in fit_file and 'FM' in fit_file:
	mode_domain = 'FM_STRICT'
elif 'O1' in fit_file:
	mode_domain = 'O1'
else:
	mode_domain = 'FM'


## NEW! force alignment of colorbars!
P_wrmse_lower_lim = 3 #5
all_wrmse_lower_lim = 3 #5

P_wrmse_upper_lim = 70 #70 #50
all_wrmse_upper_lim = 70 #70 #50


#################################################################
FeH_dict = mfl.get_FeH_dict()

#################################################################
cmap = mpl.cm.rainbow.reversed()
norm = mpl.colors.Normalize(vmin=P_wrmse_lower_lim, vmax=P_wrmse_upper_lim)
m = cm.ScalarMappable(norm=norm, cmap=cmap)


#mgrid = np.arange(0.8, 5.1, 0.1)
mgrid = np.arange(1.0, 5.1, 0.1)
z_values=np.array(list(FeH_dict.keys()))


#######################################################
#
# place a grey warning box on the heatmap
# if history_file doesn't exist
#
#######################################################

#######################################################
# remake models.dat file (false by default)
########################################################
make_models_file = False

if make_models_file:
	outf = open('models.dat',"w")
	all_models = glob.glob('../LOGS/history*drag-on_seismic_p3.data')
	for f in all_models:
		outf.write(f.split('LOGS/')[1]+'\n')
	outf.close()
	print('models.dat file recreated; exiting...\n\nset `make_models_file = False` to run visualizer normally')

	sys.exit()
#else:


all_models = open("models.dat","r").read()
#print('all_models: ', all_models)

missing_m = []
missing_z = []
missing_FeH = []
for k in mgrid:
	for l in z_values:
		test_str = 'history_m'+"%.2f"%float(k)+'_z'+"%.4f"%float(l)+'_eta0.01_drag-on_seismic_p3.data' #../LOGS/
		if test_str not in all_models:
			missing_m.append(float(k))
			missing_z.append(float(l))
			missing_FeH.append(FeH_dict["%.4f"%float(l)])
			#print('warning! ', test_str, " not found!!")

			#print('sbatch exec.slurm '+"%.2f"%float(k)+' '+"%.4f"%float(l)+ '\nsleep 60')

missing_m = np.array(missing_m)
missing_z = np.array(missing_z)
missing_FeH = np.array(missing_FeH)


#sys.exit()


########################################################
# data loading
########################################################
masses = []
zs= []
FeH=[]
pulse_num = []
P_wrmse = []
total_wrmse = []
inf = open(fit_file)
for line in inf:
	if line and "#" not in line:
		p = line.split()
		masses.append(float(p[1]))
		zs.append(float(p[2]))
		corresponding_FeH = FeH_dict["%.4f"%float(p[2])]
		FeH.append(corresponding_FeH)
		pulse_num.append(float(p[3]))
		P_wrmse.append(float(p[4]))
		total_wrmse.append(float(p[5]))
inf.close()


masses = np.array(masses)
zs = np.array(zs)
FeH = np.array(FeH)
pulse_num = np.array(pulse_num)
P_wrmse = np.array(P_wrmse)
all_wrmse = np.array(total_wrmse)

zgrid=np.array(list(FeH_dict.values()))

#mm, zz = np.meshgrid(mgrid, zgrid)

############################
#
# build null m,z vectors
#
###########################
mm = []
zz = []
for mi in mgrid:
	for zi in zgrid:
		mm.append(mi)
		zz.append(zi)
mm= np.array(mm)
zz = np.array(zz)

#print("grid size: ", len(mm))


##########################################################
#
# map Pwrmse statistic onto mass, FeH coords
#
##########################################################
local_best_fits = {}
for i in range(len(masses)):
	these_pulses = np.where( (masses[i] == masses) & (zs[i]==zs) )[0]

	if use_mixed_statistic:
		local_best_fit = all_wrmse[these_pulses].min() 
	else:
		local_best_fit = P_wrmse[these_pulses].min() 
		
	local_best_fits[(masses[i], FeH[i])]=local_best_fit



uniq_masses = []
uniq_FeH = []
for t in local_best_fits.keys():
	uniq_masses.append(t[0])
	uniq_FeH.append(t[1])
uniq_masses = np.array(uniq_masses)
uniq_FeH = np.array(uniq_FeH)

uniq_z = []
for val in uniq_FeH:
	for key, value in FeH_dict.items():
		if val == value:
			uniq_z.append(key)
uniq_z = np.array(uniq_z)

best_score = []
for d in local_best_fits.values():
	best_score.append(d)
best_score = np.array(best_score)

colors_rgba = m.to_rgba(best_score)
hex_colors = []
for cc in colors_rgba:
	hex_colors.append(mpl.colors.rgb2hex(cc))

########################################################################


################################################
## second dictionary containing figure addresses
#              https://github.com/mjoyceGR/AGB_grid_visualizer/blob/main/associated_pulse_spectra/Pvt_m0.80_FeH-0.530.png
#file_header = 'https://github.com/mjoyceGR/AGB_grid_visualizer/blob/main/'
#file_header = '/home/mpj004/meridithjoyce.com/images/AGB_grid/'

file_header_Nov9   = 'https://meridithjoyce.com/images/AGB_grid/Nov9_2023/'
file_header_Nov28  = 'https://meridithjoyce.com/images/AGB_grid/Nov28_2023/'
file_header        = 'https://meridithjoyce.com/images/AGB_grid/Dec7_2023/'+mode_domain+'/'

Z_dict   = mfl.get_Z_dict()
FeH_dict = mfl.get_FeH_dict()


######################################################################################
#
# create data object only for those models with pngs 
#
######################################################################################
png_masses = []
png_FeH=[]
png_Z = []
pngs_Nov9 = []
pngs = []
data_urls = []
png_ages=[]

mass_from_ages, z_from_ages, ages= np.loadtxt('model_ages.dat',usecols=(0,1,2), unpack=True)

#outf=open('names_of_pngs.dat','w')
#for f in glob.glob('../associated_pulse_spectra/*.png'):
#print('mode_domain: ',mode_domain)
for longf in glob.glob('peak_ID_figures/'+mode_domain+'/*.png'):
#for f in open('names_of_pngs.dat',"r").readlines():
	#outf.write(f.split('../associated_pulse_spectra/')[1]+'\n')

	f = longf.split('peak_ID_figures/'+mode_domain+'/')[1]
	#print(f)

	#############################
	#
	# from associated pulse spectra
	#
	#############################
	#pngs_Nov9.append(file_header_Nov9 + f )
	#mass_val = float(f.split('Pvt_m')[1].split('_FeH')[0])
	# png_masses.append( mass_val )
	# FeH = float( f.split('FeH')[1].split('.png')[0] )
	# png_FeH.append(FeH)
	# z_val =  Z_dict[float(FeH)]
	# png_Z.append(z_val)  ## Z value correspond to floating point FeH key

	#print('png location: ',file_header + f)
	pngs.append(     file_header + f )

	#hits_STRICT-FM_5.00_0.0344.png
	if 'FM' in mode_domain:
		mass_val = float(f.split('FM')[1].split('_')[1])
		z_val    = float(f.split('FM')[1].split('_')[2].split('.png')[0])

	elif 'O1' in mode_domain:
		mass_val = float(f.split('O1')[1].split('_')[1])
		z_val    = float(f.split('O1')[1].split('_')[2].split('.png')[0])

	png_masses.append( mass_val )
	png_Z.append(z_val)
	#FeH = float( f.split('FeH')[1].split('.png')[0] )

#	z_val =  Z_dict[float(FeH)]
	FeH = FeH_dict["%.4f"%z_val]
	png_FeH.append(FeH)

  ## Z value correspond to floating point FeH key

	## precede with "LOGS/" for local access
	hist_file_name = 'history_m'+"%.2f"%float(mass_val)+'_z'+"%.4f"%float(z_val)+'_eta0.01_drag-on_seismic_p3.data' 
	hist_url = 'https://meridithjoyce.com/pulse_data/'+hist_file_name

	try:
		assoc_age = ages[ np.where(  (mass_from_ages == float(mass_val)) & (z_from_ages == float(z_val)) )[0]  ] 
		formatted_assoc_age = "%.2f"%float(assoc_age/1.0e3)

		png_ages.append(formatted_assoc_age)
	except:
		png_ages.append('not found')	
	data_urls.append(hist_url)

#outf.close()


image_dict = {
		      'png_masses'    : png_masses,
		      'png_FeH'       : png_FeH,
		      'png_Z'         : png_Z,
		      'png_ages'      : png_ages,		      
		      'pngs' 	      : pngs,
		      # 'pngs_Nov9' 	  : pngs_Nov9,
		      'data_urls'     : data_urls
              }

image_df = pd.DataFrame(data=image_dict)
ds = ColumnDataSource(data=image_df)
ht = HoverTool()
div = Div(text="")



## my own defintiion of hover is at the top of this script
#TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,"   
p = figure(tools=["pan","wheel_zoom","zoom_in","zoom_out","box_zoom","undo","redo, reset"],\
           toolbar_location="right", width=800, height=600) #"crosshair"  # removed "ht" ht,"tap",
p.toolbar.logo = "grey"

frame1 = p.scatter(x=mm, y=zz, marker='square', fill_color='navy', size=12, alpha=1, line_width = 0) ## this covers whole layer with navy squares
frame2 = p.scatter(x=missing_m, y=missing_FeH, marker='square', fill_color='lightgrey', size=12, alpha=1, line_width = 0)
frame3 = p.scatter(x=uniq_masses, y=uniq_FeH, fill_color=hex_colors, size=16, line_width=0, marker='square')#,\


frame4 = p.scatter(source=ds , x="png_masses", y="png_FeH", color='lightgrey', alpha=0, size=16, line_width=0, marker='star')


####################################
#
# mass,Z indexing will be WRONG if the data source
# is not the correct one for the frame
# img_dict applies only to frame4
#
####################################
custom_tap = TapTool(renderers=[frame4],callback=OpenURL(url="@data_urls") )
p.add_tools(custom_tap)


########################
#
# changed indices.length > 0
# to 
# indices.length >= 0
# !!!!
# to remove the "random image substitution" issue 
#
########################
ht_callback = CustomJS(args=dict(div=div, ds=ds), code="""
    const hit_test_result = cb_data.index;
    const indices = hit_test_result.indices;
    if (indices.length >= 0) {
         div.text = `
                <img
                src="${ds.data['pngs'][indices[0]]}" height="400" alt="no pulse spectrum available"
                style="float: left; margin: 0px 15px 15px 0px; image-rendering: crisp-edges;"
                border="2"
                ></img>

				<h2>mass = ${ds.data['png_masses'][indices[0]]} Msolar
		 		</h2>
				<h2> [Fe/H] = ${ds.data['png_FeH'][indices[0]]} dex
				</h2>
				<h2> Z = ${ds.data['png_Z'][indices[0]]}
				</h2>
				<h2> median age = ${ds.data['png_ages'][indices[0]]} Gyr
				</h2>
				<p><a href=${ds.data['data_urls'][indices[0]]}> click the point to download data for this file </a></p>

                `;

    }
""")

custom_hover = HoverTool(renderers=[frame4], callback=ht_callback)
p.add_tools(custom_hover)

## the next line suppresses hover boxes appearing next to the cursor
p.hover.tooltips = [
    (""     , ""),
	]
## the labels
# p.hover.tooltips = [
#     ("mass"     , "@png_masses"),
#     ("[Fe/H]"   , "@png_FeH"),
#     ("Z"        , "@png_Z"),
#     ("Age (Gyr)", "@png_ages")
# 	]


p.title=fit_file#.split()[1]

p.xaxis.axis_label = "Model Initial Mass"
p.yaxis.axis_label = "[Fe/H] (dex)"
#p.xaxis.xlim = ()
#p.yaxis.lim = ()

#output_file("heatmap.html", title="Bokeh mass-[Fe/H] grid")#, mode='inline')

layout = column(row(p, div))
#layout = column(button,row(p, div))
show(layout)




