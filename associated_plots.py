#!/usr/bin/env python3
####################################################
#
# Author: M Joyce
#
####################################################
import numpy as np
import glob
import sys
import subprocess
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.patches as patches

import argparse

sys.path.append('../py_mesa_reader/')
import mesa_reader as mr

#---------------------------
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import Normalize
cmap = mpl.cm.gist_rainbow.reversed()
#cmap = mpl.colormaps['Greys']
norm = mpl.colors.Normalize(vmin=1.5, vmax=3.0)
m = cm.ScalarMappable(norm=norm, cmap=cmap)


#---------------------------------
parser = argparse.ArgumentParser(description='generate figure')
parser.add_argument('savefig', help='type "y" to save figure', type=str)
args = parser.parse_args()
cmdLine=True
if 'y' in args.savefig:
	savefig = True
else:
	savefig = False
#---------------------------------

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx],idx

#------------------------------

def period_in_days(f):
	#f = float(f)
	p = 1.0/(f*(1e6*0.0864)  )
	return p

def sortKeyFunc(f):
	kf = f.split('LOGS/history_m')[1].split('_z')[0]
	return kf 
#--------------------------

FeH_dict={"0.0001":-2.2,\
		  "0.0005":-1.53,\
		  "0.0010":-1.2,\
		  "0.0013":-1.105,\
		  "0.0018":-1.02,\
		  "0.0020":-0.93,\
		  "0.0025":-0.835,\
		  "0.0030":-0.75,\
		  "0.0036":-0.675,\
		  "0.0040":-0.60,\
		  "0.0050":-0.53,\
		  "0.0060":-0.45,\
		  "0.0070":-0.38,\
		  "0.0080":-0.325,\
		  "0.0095":-0.25,\
		  "0.0100":-0.20,\
		  "0.0125":-0.13,\
		  "0.0135":-0.075,\
		  "0.0140":0.00,\
		  "0.0200":0.055,\
		  "0.0216":0.11,\
		  "0.0247":0.17,\
		  "0.0300":0.25,\
		  "0.0344":0.318,\
		  "0.0400":0.39,\
		  "0.0500":0.49,\
		  "0.0600":0.57,\
		  "0.0700":0.645,\
		  "0.0800":0.70,\
		  "0.0900":0.765,\
		  "0.1000":0.82,\
		  "0.1100":0.86,\
		  "0.1200":0.905,\
		  "0.1300":0.945,\
		  "0.1400":0.98,\
		  "0.1500":1.00,\
		   }

############################################################
#
# Obs Box -- THESE ARE THE STRICT PARAMS!
#
############################################################
Teff_median = 2500 	# Kelvin
Teff_err  = 500. #*2.0	# Kelvin 

logL_median = 4 #meaning 10^4 meaning 10000
logL_err = 0.2 #*2.0

## extremely generous R bounds
Rmin = 300 #*0.7
Rmax = 550 #*1.3

Psigma = 30 ## made up!

###########################################################
#
# load observations to inherit upper and lower bounds for plot
#
###########################################################
load_file = 'r_hya_supplemented_years.csv'
yrs, obs_P_days= np.loadtxt(load_file, usecols=(0,1), unpack = True, delimiter=',')

Pmin = obs_P_days.min() - Psigma
Pmax = obs_P_days.max() + Psigma

#########################################
#for z_iter in [0.03]: # [0.0095, 0.010, 0.014, 0.020, 0.05, 0.08, 0.1, 0.15]: #:, 0.0095, 0.009]: 

	# ztag="%.4f"%z_iter 
	# feh = FeH_dict[ztag]
	# print('feh: ', feh)

tag='drag-on_seismic'
mass="2.90"#'4.0'
ztag='0.0247'

to_end = glob.glob(   'LOGS/history_m'+mass+"*_z"+ztag+'*'+tag+'_p3*.data')
MESA_models = to_end 
MESA_models.sort(key=sortKeyFunc)

#print('MESA_models: ', MESA_models)
#log = open('write_errors.log','w')
for i in range(len(MESA_models)):
	f = MESA_models[i]
	print("MESA model f: ",f)
	
	try:
		mass_tag = f.split('LOGS/history_m')[1].split('_z')[0]
		ztag = f.split('_z')[1].split('_')[0]
		feh = FeH_dict[ztag]


		label='mass='+"%.2f"%float(mass_tag)+r'$M_{\odot}$' +\
		  			  ', [Fe/H]='+str(feh) #+\
		  			  #' ' + f.split('.data')[0].split('_')[4]
		md = mr.MesaData(f)
		
		model_number = md.model_number
		star_mass = md.star_mass
		star_age = md.star_age
		log_LH = md.log_LH
		log_Teff = md.log_Teff
		log_L = md.log_L
		log_R = md.log_R
		log_g = md.log_g
		phase_of_evolution = md.phase_of_evolution
		#print("phase_of_evolution: ",phase_of_evolution)

		num_retries = md.num_retries


		#f_mode_frequency = md.nu_radial_0  
		FM_frequency = md.nu_radial_0
		O1_frequency = md.nu_radial_1

		timestep = md.log_dt
		timestep_in_years = 10.0**(timestep)  #/(525600.0*60.0)


		FM_frequency = md.nu_radial_0*(1e6*0.0864)
		O1_frequency = md.nu_radial_1*(1e6*0.0864)
	#	f_mode_frequency = (f_mode_frequency)

		#f_period = 1.0/f_mode_frequency
		FM_period = 1.0/FM_frequency
		O1_period = 1.0/O1_frequency


		Teff = 10.0**log_Teff
		star_age = star_age/1e6
		radius = 10.0**log_R


		L_Teff_domain = np.where( (  (logL_median + logL_err) > log_L) \
		 					& (  (logL_median - logL_err) < log_L) \
		 					& (  (Teff_median + Teff_err) > Teff) \
		 					& (  (Teff_median - Teff_err) < Teff) )

		L_Teff_R_domain = np.where( (  (logL_median + logL_err) > log_L) \
		 					& (  (logL_median - logL_err) < log_L) \
		 					& (  (Teff_median + Teff_err) > Teff) \
		 					& (  (Teff_median - Teff_err) < Teff) \
		 					& (  (radius) >= Rmin) \
		 					& (  (radius) <= Rmax) ) 

		R_domain = np.where(  (  (radius) >= Rmin) \
		 					& (  (radius) <= Rmax) ) 


		phase8 = np.where( phase_of_evolution == 8)[0]
		phase9 = np.where( phase_of_evolution == 9)[0]



		fig, ax = plt.subplots(figsize=(20,16))

		plt.plot(star_age, FM_period, color='black', marker='*', markersize=12, linestyle='',label='FM')
		plt.plot(star_age, O1_period, color='navy', marker='o', markersize=8, linestyle='', label='O1')

		plt.plot(0,0, color='pink', marker='D', markersize=20, alpha=0.5, linestyle='',label=r'R domain')
		plt.plot(0,0, color='yellow', marker='s', markersize=30, alpha=0.5, linestyle='',label='L-Teff domain')



		### FM
		plt.plot(star_age[R_domain], FM_period[R_domain], color='pink', marker='D', markersize=18, alpha=0.3, linestyle='')#,label=r'R domain')
		plt.plot(star_age[L_Teff_domain], FM_period[L_Teff_domain], color='yellow', marker='s', markersize=35, alpha=0.2, linestyle='')#,label='L-Teff domain')

		### first overtone
		plt.plot(star_age[R_domain], O1_period[R_domain], color='pink', marker='D', markersize=18, alpha=0.3, linestyle='')
		plt.plot(star_age[L_Teff_domain], O1_period[L_Teff_domain], color='yellow', marker='s', markersize=35, alpha=0.2, linestyle='')



		plt.xlim(xmin=star_age.min()  - 0.01, xmax=star_age.max()+0.01)
		plt.ylim(ymin=O1_period.min() - 10,\
		         ymax= max(Pmax ,min(  FM_period[np.where(FM_period < 1000)].max()+20.,  1000.)) \
		         )

		#print('components of max function: ', Pmax,  FM_period[np.where(FM_period < 1000)].max()+20., 1000. )


		ax.set_xlabel("Age (Myr)",fontsize=40)
		ax.set_ylabel(r"Period (days)", fontsize=40)

		ax.tick_params(axis='both', which='major', labelsize=26)
		ax.tick_params(axis='both', which='minor', labelsize=26)

		ax.xaxis.set_minor_locator(AutoMinorLocator())
		ax.yaxis.set_minor_locator(AutoMinorLocator())
		ax.tick_params(which='both', width=4)
		ax.tick_params(which='major', length=12)
		ax.tick_params(which='minor', length=8, color='black')


		bottom = Teff_median - Teff_err #logL_median - logL_err
		height = 2.0*Teff_err

		plt.axhline(y = obs_P_days.min(), linestyle='--', color='red', linewidth=3)
		plt.axhline(y = obs_P_days.max(), linestyle='--', color='red', linewidth=3, label='observed bounds on Period (d)')

		plt.legend(loc=2, fontsize=25)
		#plt.xlim(1268.2, star_age.max())

		text_str = label
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
		xmin, xmax= plt.gca().get_xlim()
		ymin, ymax = plt.gca().get_ylim()
		xspan = xmax - xmin
		yspan = ymax - ymin
		xval = xmin + 0.03*xspan
		yval = ymax - 0.28*yspan
		ax.text(xval, yval, text_str, fontsize=25, bbox = props)


		if savefig:
			plt.savefig('/data2/mjoyce/MESA/work_AGB_mesa-dev/associated_pulse_spectra/Pvt_m'+mass_tag+'_FeH'+"%.3f"%feh+'.png')
		else:
			plt.show()
		plt.close()

		#sys.exit()

	except:
		print("file read failure on ",f)
		#log.write(f+'\n')

#log.close()
