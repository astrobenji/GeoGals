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

DEG_PER_RADIAN = 180/np.pi

wavelengths = np.array([4861.3, 4958, 5006.0, 6548.0, 6562.0, 6583.0, 6716.0, 6731.0])

gal_IDs = ['NGC3351', 'IC5332', 'NGC1433', 'NGC4321']

cmap = mpl.colormaps.get_cmap('Blues')
cmap.set_bad(color='black')

# for gal_ID in gal_IDs:
# 	gal_df = gg.open_line_df(gal_ID)
# 	meta   = gg.meta_getter(gal_ID)

# for all gals
metadata = gg.open_metadata()

for meta in metadata:
	gal_ID = meta['Galaxy_ID']
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
	Hii_map  = np.log10(gal_df[line_IDs.index('HA6562_FLUX')].data * combo_classification)
	fig, ax = plt.subplots(figsize=(5,5))
	ax.imshow(Hii_map, cmap=cmap, extent=[np.min(RA_grid), np.max(RA_grid), np.min(DEC_grid), np.max(DEC_grid)])
	# Add ellipses
	centre  = (meta['RA'], meta['DEC'])
	delta_center = 0.00005
	r25_deg = meta['R25_kpc'] / (1000 * meta['Dist']) * DEG_PER_RADIAN
	inner_e = Ellipse(centre, width=0.4*r25_deg*np.cos(np.deg2rad(meta['i'])), height = 0.4*r25_deg, angle=-1*meta['PA'], facecolor='none', edgecolor='red')
	middl_e = Ellipse(centre, width=r25_deg*np.cos(np.deg2rad(meta['i'])), height = r25_deg, angle=-1*meta['PA'], facecolor='none', edgecolor='red', linestyle='--')
	outer_e = Ellipse(centre, width=1.6*r25_deg*np.cos(np.deg2rad(meta['i'])), height = 1.6*r25_deg, angle=-1*meta['PA'], facecolor='none', edgecolor='red')
	ax.add_patch(inner_e)
	ax.add_patch(middl_e)
	ax.add_patch(outer_e)
	ax.scatter(meta['RA'], meta['DEC'], color='red', marker='+')
	center_bounds = np.array([[meta['RA'] + delta_center, meta['DEC']],
	                 [meta['RA'] - delta_center, meta['DEC']],
	                 [meta['RA'], meta['DEC'] + delta_center],
	                 [meta['RA'], meta['DEC'] - delta_center]])
	                 
	ax.scatter(center_bounds[:,0], center_bounds[:,1], color='red', marker='o')
	
	#ax.set_xticks([]); ax.set_yticks([])
	plt.tight_layout()
	fig.savefig('../Plots/Center_Gal_Images/{0}.png'.format(gal_ID), dpi=100)
	