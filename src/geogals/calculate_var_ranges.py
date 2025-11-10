'''
What are the ranges on log(sigma^2) for observed galaxies?

Created by: Benjamin Metha
Last Updated: Sep 17, 2025
'''

import numpy as np
import GeoGals as gg
import pandas as pd

from TYPHOON_wrangling import open_metadata

diags = ['O3N2', 'Scal', 'N2S2Ha']

metadata = open_metadata()

result_holder = np.empty((len(diags), len(metadata)))

for ii, diag in enumerate(diags):
	for jj, meta in enumerate(metadata):
		gal_ID = meta['Galaxy_ID']
		# Open the emcee result
		result_df = pd.read_pickle('../Results/emcee_exp_fits/all_trained_params/subsamp/{0}_{1}.pkl'.format(gal_ID, diag))
		result_holder[ii,jj] = result_df['median']['log_Var']
		
