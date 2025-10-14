'''
plot_results.py

Plots the key parameters of interest (phi, efficiency)
for all galaxies
using a given diagnostic

Order the galaxies by mass

For phi, include the results from Comeron+18 as error bars

Created by: Benjamin Metha
Last updated: Apr 10, 2025
'''

import GeoGals as gg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plot_path = '../Plots/Results/Smallscale_params_over_galaxy_sample/phi_vs_e_'

metadata = gg.open_metadata()

gal_IDs    = [meta['Galaxy_ID'] for meta in metadata]
gal_masses = [meta['log_M_star'] for meta in metadata]
gal_D_MS   = [meta['D_SFMS'] for meta in metadata]

# order gal_IDs by their masses
gal_IDs = [x for _, x in sorted(zip(gal_masses, gal_IDs))]

y= 0.015 # yield from kobayashi...

diags = ['O3N2', 'N2S2Ha', 'Scal']

for diag in diags:
	phi_centres    = []
	phi_upper_errs = []
	phi_lower_errs = []
	
	eps_centres    = []
	eps_upper_errs = []
	eps_lower_errs = []
	for gal_ID in gal_IDs:
		# read derived properties
		result_df = pd.read_pickle('../Results/emcee_exp_fits/derived_params/subsamp/{0}_{1}.pkl'.format(gal_ID, diag))
		phi_centres.append(result_df['phi'][50]*1000)
		phi_upper_errs.append((result_df['phi'][84] - result_df['phi'][50])*1000)
		phi_lower_errs.append((result_df['phi'][50] - result_df['phi'][16])*1000)
		
		eps_centres.append(result_df['delta_Z'][50]/ y)
		eps_upper_errs.append((result_df['delta_Z'][84] - result_df['delta_Z'][50])/y)
		eps_lower_errs.append((result_df['delta_Z'][50] - result_df['delta_Z'][16])/y)
	phi_errs = np.array([phi_upper_errs, phi_lower_errs])
	eps_errs = np.array([eps_upper_errs, eps_lower_errs])
	
	print('for {0}:'.format(diag))
	print('corrcoef: {0} pc'.format(np.corrcoef(eps_centres,phi_centres)))
	fig, ax = plt.subplots(figsize=(4,4))
	ax.errorbar(eps_centres, phi_centres, xerr=eps_errs, yerr=phi_errs, fmt='o', color='black')
	#axes[0].set_xticklabels(gal_IDs)
	ax.set_xlabel('$\epsilon$')
	ax.set_ylabel('$\phi$ (pc)')
	ax.set_yscale('log')
	#axes[1].set_ylim(0,0.5)
	if diag == 'Scal':
		plt.xlim(0,1)
	plt.tight_layout()
	fig.savefig(plot_path + diag + '.png', dpi=150)