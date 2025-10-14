'''
cut_data.py

Cut Hii dataframes to inner and outer annulus

Created by: Benjamin Metha
Last updated: Apr 04, 2025
'''

import numpy as np
import GeoGals as gg

diags = ['O3N2', 'N2S2Ha', 'Scal']

n_dp_to_sample = 500

def get_subsample(n_dp, n_in_subsample):
	'''
	Creates an n_dp long array, 
	n_in_subsample of which are 1, 
	the rest are zero.
	'''
	if n_dp < n_in_subsample:
		raise ValueError
	A = np.zeros(n_dp)
	A[:n_in_subsample] = 1
	np.random.shuffle(A)
	return A

metadata = gg.open_metadata()

# list of galaxies with not enough pixels in their inner/outer regions
bad_gal_list = ['NGC0628', 'NGC1300', 'NGC1512', 'NGC3351', 'NGC3627', 'NGC4535']

for meta in metadata:
	gal_ID = meta['Galaxy_ID']
	if gal_ID in bad_gal_list:
		continue
	full_Hii_df = gg.open_Hii_df(gal_ID)
	r_list = gg.RA_DEC_to_radius(full_Hii_df['RA'], full_Hii_df['DEC'], meta)
	r25 = meta['R25_kpc']
	inner_disc_region = (r_list > 0.2*r25) * (r_list < 0.5*r25)
	outer_disc_region = (r_list > 0.5*r25) * (r_list < 0.8*r25)
	inner_Hii_df = full_Hii_df[inner_disc_region]
	outer_Hii_df = full_Hii_df[outer_disc_region]
	for diag in diags:
		wanted_inner_spaxels = ~np.isnan(inner_Hii_df['Z_'+diag]) & ~np.isinf(inner_Hii_df['Z_'+diag])
		n_dp_inner           = np.sum(wanted_inner_spaxels)
		try:
			subsamp_IDs          = get_subsample(n_dp_inner, n_dp_to_sample)
		except ValueError:
			print("It's an inner problem with {0}, {1}".format(gal_ID, diag))
			exit(0)
		inner_Hii_subsamp_df = inner_Hii_df[wanted_inner_spaxels][subsamp_IDs == 1]
		
		wanted_outer_spaxels = ~np.isnan(outer_Hii_df['Z_'+diag]) & ~np.isinf(outer_Hii_df['Z_'+diag])
		n_dp_outer           = np.sum(wanted_outer_spaxels)
		try:
			subsamp_IDs          = get_subsample(n_dp_outer, n_dp_to_sample)
		except ValueError:
			print("It's an outer problem with {0}, {1}".format(gal_ID, diag))
			exit(1)
		outer_Hii_subsamp_df = outer_Hii_df[wanted_outer_spaxels][subsamp_IDs == 1]
		# Save results
		inner_Hii_subsamp_df.to_pickle('../Data/Handmade/Hii_dataframes/inner_disc_subsamples/Z_maps_{0}_{1}.pkl'.format(gal_ID, diag))
		outer_Hii_subsamp_df.to_pickle('../Data/Handmade/Hii_dataframes/outer_disc_subsamples/Z_maps_{0}_{1}.pkl'.format(gal_ID, diag))