'''
Using WLS, 

find the matrix beta = [Z_C, gradZ]
for each galaxy
at each resolution
under each metallicity diagnostic.

These will be the initial guesses for gradZ and Zc
for each of our galaxies
during emcee.

Created by: Benjamin Metha
Last Updated: Mar 04, 2024 (to save central metallicities, not characteristic ones)
'''

import numpy as np 
from statsmodels.regression.linear_model import GLS

import GeoGals as gg

gal_names = ['IC5332', 'NGC0628', 'NGC1087', 'NGC1300', 'NGC1365', 'NGC1385', 'NGC1433', 'NGC1512', 'NGC1566', 'NGC1672', 'NGC2835', 'NGC3351', 'NGC3627', 'NGC4254', 'NGC4303', 'NGC4321', 'NGC4535', 'NGC5068', 'NGC7496']
diags = ['N2S2Ha', 'O3N2', 'Scal']

for gal_ID in gal_names:
	for diag in diags:
		# Open metallicity data
		Hii_df = gg.open_Hii_df(gal_ID)
		meta   = gg.meta_getter(gal_ID)
		r_char = 0.4*meta['R_25']
		wanted_spaxels = ~np.isnan(Hii_df['Z_'+diag]) & ~np.isinf(Hii_df['Z_'+diag])
		r   = gg.RA_DEC_to_radius(Hii_df['RA'][wanted_spaxels], Hii_df['DEC'][wanted_spaxels], meta)
		Z   = Hii_df['Z_'+diag]  [wanted_spaxels]
		e_Z = Hii_df['e_Z_'+diag][wanted_spaxels]
		# code used to get WLS gradients
		covariates = np.array([np.ones(len(r)), r-r_char]).T 
		Z_grad_model = GLS(Z, covariates, sigma=e_Z).fit()
		# Save np array to folder 
		np.save('../Data/Zc_gradZ_ICs/{0}_{1}.npy'.format(gal_ID, diag), Z_grad_model.params)
		np.save('../Data/Zc_gradZ_ICs/Covariances/{0}_{1}.npy'.format(gal_ID, diag), Z_grad_model.normalized_cov_params)