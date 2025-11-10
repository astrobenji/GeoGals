'''
make_subsamp_IDs.py

Lets me know which spaxels in which galaxy are selected for lower n_dp runs. 

Created by: Benjamin Metha	
Last Updated: Apr 01, 2025
'''
import numpy as np
import GeoGals as gg

np.random.seed(681260)

diags = ['N2S2Ha', 'O3N2', 'Scal']

n_dp_to_sample = 500

metadata = gg.open_metadata()

out_path  = '../Data/Handmade/subsamp_IDs/'

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

for meta in metadata:
	gal_ID = meta['Galaxy_ID']
	Hii_df = gg.open_Hii_df(gal_ID)
	meta   = gg.meta_getter(gal_ID)
	for diag in diags:
		wanted_spaxels = ~np.isnan(Hii_df['Z_'+diag]) & ~np.isinf(Hii_df['Z_'+diag])
		n_dp = np.sum(wanted_spaxels)
		subsamp_IDs = get_subsample(n_dp, n_dp_to_sample)
		np.save(out_path + '{0}_{1}_n={2}.npy'.format(gal_ID, diag, n_dp_to_sample), subsamp_IDs)