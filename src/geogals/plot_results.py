'''
plot_results.py

Plots the key parameters of interest (phi, efficiency)
for all galaxies
using a given diagnostic

Order the galaxies by mass

For phi, include the results from Comeron+18 as error bars

Created by: Benjamin Metha
Last updated: Jul 30, 2025 (to make logA figs.)
'''

import GeoGals as gg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plot_path = '../Plots/Results/Smallscale_params_over_galaxy_sample/'

# percentile data from Comeron+18:
# https://ui.adsabs.harvard.edu/abs/2018A%26A...610A...5C/abstract
comeron_phi_2low  =  62
comeron_phi_1low  = 112
comeron_phi_mid   = 191
comeron_phi_1high = 314
comeron_phi_2high = 490

metadata = gg.open_metadata()

gal_IDs    = [meta['Galaxy_ID'] for meta in metadata]
gal_masses = [meta['log_M_star'] for meta in metadata]
gal_D_MS   = [meta['D_SFMS'] for meta in metadata]

# order gal_IDs by their masses
gal_IDs = [x for _, x in sorted(zip(gal_masses, gal_IDs))]

y= 0.015 # yield from kobayashi...

diags = ['O3N2', 'N2S2Ha', 'Scal']

if __name__ != '__main__':
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
		print('median phi: {0} pc'.format(np.median(phi_centres)))
		print('median_eps: {0}'.format(np.median(eps_centres)))
		fig, axes = plt.subplots(2,1, sharex=True, figsize= (8.5,4.5))
		x_positions = np.arange(len(gal_IDs)) 
		x_low  = -0.5
		x_high = x_positions[-1] + 0.5
		axes[0].errorbar(x_positions, phi_centres, yerr=phi_errs, fmt='o', color='red', capsize=10)
		axes[1].set_xticks(x_positions, labels= gal_IDs, rotation=30)
		#axes[0].set_xticklabels(gal_IDs)
		axes[0].set_ylabel('$\\phi$ (pc)')
		#axes[0].set_title(diag)
		# Add Comeron+18 lines
		axes[0].hlines(comeron_phi_mid, xmin=x_low, xmax=x_high, color='black', linestyle="--")
		axes[0].fill_between([x_low, x_high], comeron_phi_1low, comeron_phi_1high,color='black', alpha=0.2)
		axes[0].fill_between([x_low, x_high], comeron_phi_2low, comeron_phi_2high,color='black', alpha=0.2)
		axes[0].set_xlim(x_low, x_high)
		axes[0].set_yscale('log')
		#axes[0].set_ylim(0,1000)
		
		axes[1].errorbar(x_positions, eps_centres, yerr=eps_errs, fmt='o', color='blue', capsize=10)
		axes[1].set_ylabel('$\\epsilon$')
		axes[1].set_yscale('log')
		#axes[1].set_ylim(0,0.5)
		
		plt.tight_layout()
		fig.savefig(plot_path + diag+'_smallscale_params_loglog.png', dpi=200)
	
	
# This version uses derived properties
if __name__ == '__main__':
	for diag in diags:
		logA_centres    = []
		logA_upper_errs = []
		logA_lower_errs = []
		for gal_ID in gal_IDs:
			# read derived properties
			result_df = pd.read_pickle('../Results/emcee_exp_fits/all_trained_params/subsamp/{0}_{1}.pkl'.format(gal_ID, diag))
			
			logA_centres.append(result_df['median']['log_Var'])
			logA_upper_errs.append(result_df['percentile_84']['log_Var'] - result_df['median']['log_Var'])
			logA_lower_errs.append(result_df['median']['log_Var'] - result_df['percentile_16']['log_Var'])
		logA_errs = np.array([logA_upper_errs, logA_lower_errs])
		
		fig, ax = plt.subplots()
		x_positions = np.arange(len(gal_IDs)) 
		x_low  = -0.5
		x_high = x_positions[-1] + 0.5
		ax.errorbar(x_positions, logA_centres, yerr=logA_errs, fmt='o', color='red', capsize=10)
		ax.set_xticks(x_positions, labels = gal_IDs, rotation=30)
		ax.set_ylabel('log($\\sigma^2$)')
		ax.set_xlim(x_low, x_high)
		plt.tight_layout()
		fig.savefig(plot_path + diag+'_logA.png', dpi=200)