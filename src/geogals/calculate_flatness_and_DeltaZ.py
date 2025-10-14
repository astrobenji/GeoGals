'''
Once we have a nice set of processed emcee backends
use that data + the metadata file
to estimate (with uncertainties):

1) The flatness of a galaxy expected if R_bubble = h

2) The increase in metallicity in a 1 sigma bubble

Created by: Benjamin Metha
Last Updated: Feb 25, 2025
'''

import numpy as np
import GeoGals as gg

# Asplund+21
solar_Z_obs_units   = 8.69   # log([O/H]) + 12
solar_Z_theor_units = 0.0139 # mass fraction of heavy elements

for diag in diags:
	for gal_ID in gal_IDs:
		# Open the emcee result
		result_df = pd.read_pickle('../Results/emcee_exp_fits/{0}_{1}.pkl'.format(gal_ID, diag)
		# Open galaxy metadata
		meta = gg.meta_getter(gal_ID)
		# Compare the sizes of fluctuations to the radius (R_25) of the galaxy
		flatness_med  = result_df['median']['phi_kpc'] / meta['R25_kpc']
		flatness_low  = result_df['percentile_16']['phi_kpc'] / meta['R25_kpc']
		flatness_high = result_df['percentile_84']['phi_kpc'] / meta['R25_kpc']
		# Get the absolute size of the metallicity fluctuations
		Delta_Z_dex_med  = np.sqrt(10**result_df['median']['log_var'])
		Delta_Z_dex_low  = np.sqrt(10**result_df['percentile_16']['log_var'])
		Delta_Z_dex_high = np.sqrt(10**result_df['percentile_84']['log_var'])