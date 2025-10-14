'''
quadruplot_inner_v_outer.py

Make a 2x2 grid of plots showing inner v outer properties for Phi, DeltaZ, GradZ, Z_char

Created by: Benjamin Metha
Last updated: Mar 27, 2025
'''

# Read emcee results

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plot_path = '../Plots/Results/Split_data_comparisons/Parameters/'

for gal_ID in ['NGC4321']:
	for diag in ['N2S2Ha']:#, "N2S2Ha", 'Scal']:
		# Load in results
		inner_global_results = pd.read_pickle('../Results/emcee_exp_fits/all_trained_params/{0}_{1}_inner.pkl'.format(gal_ID, diag))
		inner_local_results  = pd.read_pickle('../Results/emcee_exp_fits/derived_params/{0}_{1}_inner.pkl'.format(gal_ID, diag))
		outer_global_results = pd.read_pickle('../Results/emcee_exp_fits/all_trained_params/{0}_{1}_outer.pkl'.format(gal_ID, diag))
		outer_local_results  = pd.read_pickle('../Results/emcee_exp_fits/derived_params/{0}_{1}_outer.pkl'.format(gal_ID, diag))
		# Plots!
		fig, axes = plt.subplots(2,2, sharex=True, figsize= (8,4.5))
		#### Ax 0,0 is Phi ######
		phi_loc = (0, 0)
		
		phi_lower_inner  = inner_local_results['phi'][16]
		phi_median_inner = inner_local_results['phi'][50]
		phi_upper_inner  = inner_local_results['phi'][84]
		
		phi_lower_outer  = outer_local_results['phi'][16]
		phi_median_outer = outer_local_results['phi'][50]
		phi_upper_outer  = outer_local_results['phi'][84]
		# Compute asymmetric error bars
		errors_inner = [[phi_median_inner - phi_lower_inner], [phi_upper_inner - phi_median_inner]]
		errors_outer = [[phi_median_outer - phi_lower_outer], [phi_upper_outer - phi_median_outer]]
		
		# X-axis positions (categorical)
		x_labels = ["inner", "outer"]
		x_positions = np.arange(len(x_labels)) 
		axes[phi_loc].errorbar(x_positions[0], phi_median_inner, yerr=errors_inner, fmt='o', color='blue', capsize=15)
		axes[phi_loc].errorbar(x_positions[1], phi_median_outer, yerr=errors_outer, fmt='o', color='blue', capsize=15)
		
		axes[phi_loc].set_xticks(x_positions)
		axes[phi_loc].set_xticklabels(x_labels)
		axes[phi_loc].set_xlim(-0.5,1.5)
		axes[phi_loc].set_ylabel("$\phi$ (kpc)")
		
		#### Ax 0,1 is DeltaZ ####
		delta_Z_loc = (0,1)
		
		delta_Z_lower_inner  = inner_local_results['delta_Z'][16]
		delta_Z_median_inner = inner_local_results['delta_Z'][50]
		delta_Z_upper_inner  = inner_local_results['delta_Z'][84]
		
		delta_Z_lower_outer  = outer_local_results['delta_Z'][16]
		delta_Z_median_outer = outer_local_results['delta_Z'][50]
		delta_Z_upper_outer  = outer_local_results['delta_Z'][84]
		# Compute asymmetric error bars
		errors_inner = [[delta_Z_median_inner - delta_Z_lower_inner], [delta_Z_upper_inner - delta_Z_median_inner]]
		errors_outer = [[delta_Z_median_outer - delta_Z_lower_outer], [delta_Z_upper_outer - delta_Z_median_outer]]
		
		# Plot
		axes[delta_Z_loc].errorbar(x_positions[0], delta_Z_median_inner, yerr=errors_inner, fmt='o', color='blue', capsize=15)
		axes[delta_Z_loc].errorbar(x_positions[1], delta_Z_median_outer, yerr=errors_outer, fmt='o', color='blue', capsize=15)
		axes[delta_Z_loc].set_ylabel("$\Delta Z$ (unitless)")
		
		#### Ax 1,0 is gradZ #####
		grad_Z_loc = (1,0)
		
		grad_Z_lower_inner  = inner_global_results['percentile_16']['grad_Z_per_kpc']
		grad_Z_median_inner = inner_global_results['median']['grad_Z_per_kpc']
		grad_Z_upper_inner  = inner_global_results['percentile_84']['grad_Z_per_kpc']
		
		grad_Z_lower_outer  = outer_global_results['percentile_16']['grad_Z_per_kpc']
		grad_Z_median_outer = outer_global_results['median']['grad_Z_per_kpc']
		grad_Z_upper_outer  = outer_global_results['percentile_84']['grad_Z_per_kpc']
		# Compute asymmetric error bars
		errors_inner = [[grad_Z_median_inner - grad_Z_lower_inner], [grad_Z_upper_inner - grad_Z_median_inner]]
		errors_outer = [[grad_Z_median_outer - grad_Z_lower_outer], [grad_Z_upper_outer - grad_Z_median_outer]]
		
		# Plot
		axes[grad_Z_loc].errorbar(x_positions[0], grad_Z_median_inner, yerr=errors_inner, fmt='o', color='blue', capsize=15)
		axes[grad_Z_loc].errorbar(x_positions[1], grad_Z_median_outer, yerr=errors_outer, fmt='o', color='blue', capsize=15)
		axes[grad_Z_loc].set_ylabel("$\\nabla Z$ (dex kpc$^{-1}$)")
		
		#### Ax 1,1 is Z_char ####
		Z_char_loc = (1,1)
		
		Z_char_lower_inner  = inner_global_results['percentile_16']['Z_char']
		Z_char_median_inner = inner_global_results['median']['Z_char']
		Z_char_upper_inner  = inner_global_results['percentile_84']['Z_char']
		
		Z_char_lower_outer  = outer_global_results['percentile_16']['Z_char']
		Z_char_median_outer = outer_global_results['median']['Z_char']
		Z_char_upper_outer  = outer_global_results['percentile_84']['Z_char']
		# Compute asymmetric error bars
		errors_inner = [[Z_char_median_inner - Z_char_lower_inner], [Z_char_upper_inner - Z_char_median_inner]]
		errors_outer = [[Z_char_median_outer - Z_char_lower_outer], [Z_char_upper_outer - Z_char_median_outer]]
		
		# Plot
		axes[Z_char_loc].errorbar(x_positions[0], Z_char_median_inner, yerr=errors_inner, fmt='o', color='blue', capsize=15)
		axes[Z_char_loc].errorbar(x_positions[1], Z_char_median_outer, yerr=errors_outer, fmt='o', color='blue', capsize=15)
		axes[Z_char_loc].set_ylabel("$Z_{char}$ (log[(O/H)] + 12)")
		
		#### Finishing touches ##
		fig.suptitle("{0} ({1})".format(gal_ID, diag), fontsize=16)
		plt.tight_layout()
		fig.savefig(plot_path + "{0}_{1}".format(gal_ID, diag), dpi=100)
	
	