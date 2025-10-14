'''
epsilon_vs_sfr.py

Plot how the star formation efficiencies in superbubble progenitor clouds relate
to the overall star formation rates of galaxies/specific star formation rates.

Prediction: sSFR will be the most strongly correlated with epsilon.

Or: if not, then star formation is happening in the same way on a local level in 
all kinds of galaxies -- same size clouds, but different cloud frequencies. I 
actually like this prediction better.

Benji Metha
April 8, 2025
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import GeoGals as gg

diags = ['O3N2', 'N2S2Ha', 'Scal']

y= 0.015 # yield from Kobayashi+20

plot_path = '../Plots/Results/smallscale_vs_large_scale/log_epsilon_vs/'

metadata = gg.open_metadata()

log_M   = np.zeros(len(metadata))
log_SFR = np.zeros(len(metadata))
D_SFMS  = np.zeros(len(metadata))


for diag in diags:
	eps_centres    = []
	eps_upper_errs = []
	eps_lower_errs = []
	print('\nFor '+diag)
	for ii, meta in enumerate(metadata):
		gal_ID = meta['Galaxy_ID']
		log_M[ii]    = meta['log_M_star']
		log_SFR[ii]  = meta['log_SFR']
		D_SFMS[ii]   = meta['D_SFMS']
		# Load in the epsilons
		result_df = pd.read_pickle('../Results/emcee_exp_fits/derived_params/subsamp/{0}_{1}.pkl'.format(gal_ID, diag))
		# Unpack them
		# Get their error bars
		eps_centres.append(result_df['delta_Z'][50]/ y)
		eps_upper_errs.append((result_df['delta_Z'][84] - result_df['delta_Z'][50])/y)
		eps_lower_errs.append((result_df['delta_Z'][50] - result_df['delta_Z'][16])/y)
		# Make plots with each dep. variable
	eps_errs = np.array([eps_upper_errs, eps_lower_errs])
	
	log_sSFR = log_SFR - log_M
	# Just mass
	fig, ax = plt.subplots(figsize=(5,4))
	x_ax = log_M
	x_label = 'log($M_* / M_\odot$)'
	ax.errorbar(x_ax, eps_centres, yerr=eps_errs, fmt='o', color='blue', capsize=10)
	ax.set_ylabel('$\epsilon$')
	ax.set_xlabel(x_label)
	ax.set_yscale('log')
	plt.tight_layout()
	fig.savefig(plot_path + 'eps_vs_M_{0}.png'.format(diag), dpi=200)

	
	x_axes   = [log_M, log_SFR, D_SFMS, log_sSFR]
	x_labels = ['log_M*', 'log_SFR', 'D_SFMS', 'log_sSFR']
	for x_ax, x_label in zip(x_axes, x_labels):
		fig, ax = plt.subplots()
		ax.errorbar(x_ax, eps_centres, yerr=eps_errs, fmt='o', color='red', capsize=10)
		ax.set_ylabel('$\epsilon$')
		ax.set_xlabel(x_label)
		ax.set_yscale('log')
		plt.tight_layout()
		corr = np.corrcoef(x_ax, np.log10(eps_centres))
		print(f'corr(log_eps, {x_label}) = {corr}')
		fig.savefig(plot_path + '{0}_{1}.png'.format(x_label, diag), dpi=100)