'''
inner_outer_ellipse_plot.py

Plot an ellipse that visualises where the inner and outer disc regions of the
galaxies start and end.

Created by: Benjamin Metha
Last updated: Mar 27, 2025
'''
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse

import numpy as np

import GeoGals as gg

from Z_diags import *

DEG_PER_RADIAN = 180/np.pi

wavelengths = np.array([4861.3, 4958, 5006.0, 6548.0, 6562.0, 6583.0, 6716.0, 6731.0])

cmap = mpl.colormaps.get_cmap('viridis')
cmap.set_bad(color='black')

def untwist(twisted_XY, PA):
	PA = PA*np.pi/180
	init_X  = twisted_XY[:,0]
	init_Y  = twisted_XY[:,1]
	final_X = init_X*np.cos(PA) + init_Y*np.sin(PA)
	final_Y = init_Y*np.cos(PA) - init_X*np.sin(PA)
	return final_X, final_Y

# for gal_ID in gal_IDs:
# 	gal_df = gg.open_line_df(gal_ID)
# 	meta   = gg.meta_getter(gal_ID)

# for all gals
metadata = gg.open_metadata()

for meta in metadata:
	gal_ID = meta['Galaxy_ID']
	if gal_ID != 'NGC1385':
		continue 
	gal_df = gg.open_line_df(gal_ID)
	meta   = gg.meta_getter(gal_ID)
	# preprocessing
	RA_grid, DEC_grid = gg.make_RA_DEC_grid(gal_df[0].header)
	# replace lines where S/N < 10 with nans
	gg.SN_cut(gal_df, 10) 
	gg.extinction_correction(gal_df, wavelengths)
	# BPT diagnostics
	S2_BPT_classification = gg.classify_S2_BPT(gal_df)
	N2_BPT_classification = gg.classify_N2_BPT(gal_df)
	combo_classification  = S2_BPT_classification*N2_BPT_classification
	line_IDs = [gal_df[x].header['EXTNAME'] for x in range(len(gal_df))]
	Z_O3N2, e_Z_O3N2	  = compute_Z_O3N2_Curti17(gal_df)
	Z_O3N2[combo_classification == 0] = np.nan
	# Get X, Y limits
	AB_grid  = gg.RA_DEC_to_XY(RA_grid, DEC_grid, meta)
	X_grid, Y_grid = untwist(AB_grid, meta['PA'])
	X_tix = np.linspace(np.min(X_grid), np.max(X_grid), 9)
	Y_tix = np.linspace(np.min(Y_grid), np.max(Y_grid), 9)
	# Plot!
	fig, ax = plt.subplots(figsize=(5,5))
	im = ax.imshow(Z_O3N2, cmap=cmap, extent=[np.min(X_tix), np.max(X_tix), np.min(Y_tix), np.max(Y_tix)])
	plt.colorbar(im, label='log([O/H])+12')
	ax.set_xlabel('X (kpc)')
	ax.set_ylabel('Y (kpc)')
	
	# Code for plotting Ha map:
# 	log_Ha = np.log10(gal_df[29].data) - 20
# 	fig, ax = plt.subplots(figsize=(5,5))
# 	im = ax.imshow(log_Ha, cmap='Blues', extent=[np.min(X_tix), np.max(X_tix), np.min(Y_tix), np.max(Y_tix)])
# 	ax.set_facecolor('black')
# 	plt.colorbar(im, label='log(erg/sec/cm$^2$/spaxel)')
# 	ax.set_xticks(X_tix)
# 	ax.set_yticks(Y_tix)
	plt.title(gal_ID)
	plt.show()
	
	# Add ellipses
# 	centre  = (meta['RA'], meta['DEC'])
# 	delta_center = 0.00005
# 	r25_deg = meta['R25_kpc'] / (1000 * meta['Dist']) * DEG_PER_RADIAN
# 	inner_e = Ellipse(centre, width=0.4*r25_deg*np.cos(np.deg2rad(meta['i'])), height = 0.4*r25_deg, angle=-1*meta['PA'], facecolor='none', edgecolor='red')
# 	middl_e = Ellipse(centre, width=r25_deg*np.cos(np.deg2rad(meta['i'])), height = r25_deg, angle=-1*meta['PA'], facecolor='none', edgecolor='red', linestyle='--')
# 	outer_e = Ellipse(centre, width=1.6*r25_deg*np.cos(np.deg2rad(meta['i'])), height = 1.6*r25_deg, angle=-1*meta['PA'], facecolor='none', edgecolor='red')
# 	ax.add_patch(inner_e)
# 	ax.add_patch(middl_e)
# 	ax.add_patch(outer_e)
# 	ax.scatter(meta['RA'], meta['DEC'], color='red', marker='+')
# 	center_bounds = np.array([[meta['RA'] + delta_center, meta['DEC']],
# 	                 [meta['RA'] - delta_center, meta['DEC']],
# 	                 [meta['RA'], meta['DEC'] + delta_center],
# 	                 [meta['RA'], meta['DEC'] - delta_center]])
# 	                 
# 	ax.scatter(center_bounds[:,0], center_bounds[:,1], color='red', marker='o')
# 	
	#ax.set_xticks([]); ax.set_yticks([])
#  	plt.tight_layout()
#  	fig.savefig('../Plots/Figure2.png'.format(gal_ID), dpi=200)
	