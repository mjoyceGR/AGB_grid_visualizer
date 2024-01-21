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
parser.add_argument('use_mixed_statistic', help='\nIf n, use period only. '+\
	                                            '\nIf y, use P+LTR pseudo-chisq (recommended).'+\
	                                            '\nIf L, use L_w.'+\
	                                            '\nIf T, use T_w.'+\
	                                            '\nIf R, use R_w.'+\
	  	                                        '\nIf age, use age.'+\
	  	                                        '\nIf pulse_num, use best pulse_num according to "y" statistic.'+\
	                                            '\nIf H, use harmonic mean of P,LTR.', type=str)
args = parser.parse_args()
cmdLine=True

if args.use_mixed_statistic == 'H':
	use_alt = False
else:
	use_alt = True

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


# ## NEW! force alignment of colorbars!
# P_wrmse_lower_lim = 3 #5
# all_wrmse_lower_lim = 3 #5

# P_wrmse_upper_lim = 70 #70 #50
# all_wrmse_upper_lim = 70 #70 #50


#################################################################
FeH_dict = mfl.get_FeH_dict()

#print('FeH_dict: ', FeH_dict)

#################################################################

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
make_models_file = True
use_He = False

if make_models_file:

	if use_He:
		outf = open('varied_Yi_models.dat',"w")
		all_models = glob.glob('../LOGS/history*yi-on_seismic_p3.data')
	
	else:
		outf = open('fixed_Yi_models.dat',"w")
		all_models    = glob.glob('LOGS/history*drag-on_seismic_p3.data')
	

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
		test_str = 'history_m'+"%.2f"%float(k)+'_z'+"%.4f"%float(l)+'_eta0.01_drag-on_seismic_p3.data' 
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
L_w = []
T_w = []
R_w = []
total_wrmse = []
median_age = []
##Period    mass   z  abs_pulse_number  period_WRMSE  Lw   Tw   Rw   global_WRMSE   median_age
## 0         1     2         3               4         5    6   7          8           9
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
		#try:
		L_w.append(float(p[5]))
		T_w.append(float(p[6]))
		R_w.append(float(p[7]))
		total_wrmse.append(float(p[8]))
		median_age.append(float(p[9]))
		# except IndexError:
		# 	total_wrmse.append(float(p[5]))
inf.close()


masses = np.array(masses)
zs = np.array(zs)
FeH = np.array(FeH)
pulse_num = np.array(pulse_num)
P_wrmse = np.array(P_wrmse)
all_wrmse = np.array(total_wrmse)

#try:
L_w = np.array(L_w)
T_w = np.array(T_w)
R_w = np.array(R_w)

alt_Hw = np.sqrt( (1.0/3.0)*(L_w**2.0 + T_w**2.0 + R_w**2.0) + P_wrmse**2.0 ) 

median_age = np.array(median_age)
pulse_num = np.array(pulse_num)

# except:
# 	pass

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

##################### choose which statistic ##############################
local_best_fits = {}
#	P_wrmse_lower_lim = 5 #best_score.min()

for i in range(len(masses)):
	these_pulses = np.where( (masses[i] == masses) & (zs[i]==zs) )[0]

	if args.use_mixed_statistic == 'y':
		if use_alt:
			local_best_fit = alt_Hw[these_pulses].min() 
			tag = 'altH' 
			formatted_tag = r'$\sqrt{ \frac{1}{3}(L_w^2 + T_w^2 + R_w^2) + P_w^2 }$'
			P_wrmse_upper_lim = alt_Hw.max()
			P_wrmse_lower_lim = 0

		else:
			local_best_fit = all_wrmse[these_pulses].min()
			tag = 'Hw' 
			formatted_tag = r'$H_{w}$'
			P_wrmse_upper_lim = 100
			P_wrmse_lower_lim = 5


	elif args.use_mixed_statistic == 'L':
		local_best_fit = L_w[these_pulses].min()
		tag = 'Lw' 
		formatted_tag = r'$L_{w}$'
		P_wrmse_upper_lim = 200
		P_wrmse_lower_lim = 10

	elif args.use_mixed_statistic == 'T':
		local_best_fit = T_w[these_pulses].min()
		tag = 'Tw' 
		formatted_tag = r'$T_{w}$'
		P_wrmse_upper_lim = 300
		P_wrmse_lower_lim = 10

	elif args.use_mixed_statistic == 'R':
		local_best_fit = R_w[these_pulses].min()
		tag = 'Rw' 	
		formatted_tag = r'$R_{w}$'
		P_wrmse_upper_lim = 100
		P_wrmse_lower_lim = 5

	elif args.use_mixed_statistic == 'age':
		local_best_fit =np.log10(np.median(median_age[these_pulses]))
		#print("age map: ", local_best_fit)
		tag = 'age' 	
		formatted_tag = r'log10[Age (Myr)]'
		P_wrmse_lower_lim = 2.5
		P_wrmse_upper_lim = 3.8

	elif args.use_mixed_statistic == 'pulse_num':
		#local_best_fit_ = alt_Hw[these_pulses].min() 
		this_array = np.where( alt_Hw[these_pulses].min() == alt_Hw[these_pulses] )
		
		local_best_fit = pulse_num[these_pulses][this_array][0]
		
		#print("pulse index map: ", local_best_fit)
		tag = 'pulse_num' 	
		formatted_tag = r'Pulse Index'
		P_wrmse_lower_lim = 1
		P_wrmse_upper_lim = 18

	else:
		local_best_fit = P_wrmse[these_pulses].min() 
		tag = 'Pw' 
		formatted_tag = r'$P_{w}$'
		P_wrmse_upper_lim = 70 #300
		P_wrmse_lower_lim = 5
	
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

##########################
# invert the colormap if using pulse index as colorbar
##########################
if args.use_mixed_statistic == 'pulse_num':
	cmap = mpl.cm.rainbow
else:
	cmap = mpl.cm.rainbow.reversed()
norm = mpl.colors.Normalize(vmin=P_wrmse_lower_lim, vmax=P_wrmse_upper_lim)
m = cm.ScalarMappable(norm=norm, cmap=cmap)

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

# file_header_Nov9   = 'https://meridithjoyce.com/images/AGB_grid/Nov9_2023/'
# file_header_Nov28  = 'https://meridithjoyce.com/images/AGB_grid/Nov28_2023/'
# file_header_Dec7   = 'https://meridithjoyce.com/images/AGB_grid/Dec7_2023/'+mode_domain+'/'
# file_header        = 'https://meridithjoyce.com/images/AGB_grid/Dec11_2023/'+mode_domain+'/'

							 #https://meridithjoyce.com/images/AGB_grid/Dec19_fixed_Yi/hardness100/FM/
hardness = 75
#which_grid = 'Jan17_2024'
which_grid = 'Dec19_fixed_Yi'

#mode_domain = 'FM'

website_file_header        = 'https://meridithjoyce.com/images/AGB_grid/'+which_grid+\
										'/hardness'+str(hardness)+'/'+mode_domain+'/'


image_source_location_header = '/home/mjoyce/MESA/work_AGB_mesa-dev/AGB_grid_visualizer/'
image_source_location      = 'associated_pulse_spectra/peak_detections/'+which_grid+\
										'/hardness'+str(hardness)+'/'+mode_domain+'/'


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
#pngs_Nov9 = []
pngs = []
data_urls = []

png_ages=[]
png_pid = []




for longf in glob.glob(image_source_location+'hits_'+mode_domain+'*.png'):
	#print(longf)

	f = longf.split('hardness75/'+mode_domain+'/')[1]
	#print(f)

	#############################
	#
	# from associated pulse spectra
	#
	#############################
	pngs.append(website_file_header + f )

	#hits_STRICT-FM_5.00_0.0344.png
	if 'FM' in mode_domain:
		mass_val = float(f.split('_FM')[1].split('_')[1])
		z_val    = float(f.split('_FM')[1].split('_')[2].split('.png')[0])

		#print('mass_val: ', mass_val)

	elif 'O1' in mode_domain:
		mass_val = float(f.split('_O1')[1].split('_')[1])
		z_val    = float(f.split('_O1')[1].split('_')[2].split('.png')[0])

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
	
	############# need to select the BEST from arrays with more than one option, as with PID mask above
	these_pulses = np.where(  (masses == float(mass_val)) & (zs == float(z_val)) )

	try:	
		this_pulse = np.where( alt_Hw[these_pulses].min() == alt_Hw[these_pulses] )	
		assoc_age = median_age[these_pulses][this_pulse][0]
		#print("assoc_age: ", assoc_age)
		
		formatted_assoc_age = "%.2f"%float(assoc_age/1.0e3)
		png_ages.append(formatted_assoc_age)

	except ValueError:
		png_ages.append('no match')


	try:
		this_pulse = np.where( alt_Hw[these_pulses].min() == alt_Hw[these_pulses] )
		assoc_pid = pulse_num[these_pulses][this_pulse][0]
		formatted_pid= "%.0f"%float(assoc_pid)
		png_pid.append(formatted_pid)

		############# need to select the BEST from arrays with more than one option, as with PID mask above

	except ValueError:
	 	png_pid.append('no match')	


	data_urls.append(hist_url)

#outf.close()


image_dict = {
		      'png_masses'    : png_masses,
		      'png_FeH'       : png_FeH,
		      'png_Z'         : png_Z,
		      'png_ages'      : png_ages,
		      'png_pid'       : png_pid,	      
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
           toolbar_location="right", width=700, height=550) #"crosshair"  # removed "ht" ht,"tap",
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
				<h2> best-fitting pulse index = ${ds.data['png_pid'][indices[0]]} 
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




