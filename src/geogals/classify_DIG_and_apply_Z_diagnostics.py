'''
classify_DIG_and_apply_Z_diagnostics.py

Data flow to convert emission line data into metallicity data.
For each galaxy in our subsample:

* apply SN cut
* apply an extinction correction
* Using two BPT diagrams, classify spaxels into star forming or not.
* Compute 3 different metallicity diagnostics and their associated errors
* Store all of this data in a pickled pandas df

Created by: Benjamin Metha
Last updated: Feb 20, 2025 
'''

import GeoGals as gg
import pandas as pd
import numpy as np
import logging
import os

from Z_diags import *

out_path = '/data/projects/punim1156/PHANGS/Handmade/'
wavelengths = np.array([4861.3, 4958, 5006.0, 6548.0, 6562.0, 6583.0, 6716.0, 6731.0])

# For alt galaxies:
# wavelengths = np.array([3726.0, 3729.0, 4861.3, 5007.0, 6562.8, 6583.0, 6716.0, 6731.0])
# HALPHA = 8 # index of H alpha in line dfs

if __name__=='__main__':
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
		logging.info("Running BPT diagnostics...")
		S2_BPT_classification = gg.classify_S2_BPT(gal_df)
		N2_BPT_classification = gg.classify_N2_BPT(gal_df)
		combo_classification  = S2_BPT_classification*N2_BPT_classification
		# Some standard metallicity diagnostics
		logging.info("Crafting metallicity maps...")
		Z_N2S2Ha, e_Z_N2S2Ha  = compute_Z_N2S2Ha_Dop16(gal_df)
		Z_O3N2, e_Z_O3N2	  = compute_Z_O3N2_Curti17(gal_df)
		Z_Scal, e_Z_Scal      = compute_Z_Scal_Dop16(gal_df)
		# Save Ha for computing SFR variations
		line_IDs = [gal_df[x].header['EXTNAME'] for x in range(len(gal_df))]
		Ha_index = line_IDs.index('HA6562_FLUX')
		log_Ha   = np.log10(gal_df[Ha_index].data)
		e_log_Ha = gal_df[Ha_index+1].data / gal_df[Ha_index].data
		# package this data into a pandas df
		logging.info("Packaging data products...")
		data_dict = {
		'RA':		  RA_grid.flatten(),
		'DEC':		  DEC_grid.flatten(),
		'Hii_combo':  combo_classification.flatten(),
		'Z_N2S2Ha':	  Z_N2S2Ha.flatten(),
		'e_Z_N2S2Ha': e_Z_N2S2Ha.flatten(),
		'Z_O3N2':	  Z_O3N2.flatten(),
		'e_Z_O3N2':	  e_Z_O3N2.flatten(),
		'Z_Scal':	  Z_Scal.flatten(),
		'e_Z_Scal':	  e_Z_Scal.flatten(),
		'log_Ha':     log_Ha.flatten(),
		'e_log_Ha':   e_log_Ha.flatten()
		}
		# save the combo'd Hii spaxel df
		result_df = pd.DataFrame(data_dict)
		Hii_df = result_df[result_df['Hii_combo']]
		if not os.path.exists('../Data/Handmade/Hii_dataframes/'):
			os.makedirs('../Data/Handmade/Hii_dataframes/')
		Hii_df.to_pickle('../Data/Handmade/Hii_dataframes/Z_maps_{0}.pkl'.format(gal_ID))
		logging.info("OK.\n")
