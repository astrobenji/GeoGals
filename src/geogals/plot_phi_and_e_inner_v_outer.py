'''
quadruplot_inner_v_outer.py

Make a 2x2 grid of plots showing inner v outer properties for Phi, DeltaZ, GradZ, Z_char

Created by: Benjamin Metha
Last updated: Apr 09, 2025
'''

# Read emcee results

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plot_path = '../Plots/Results/'

split_gals = ['IC5332', 'NGC1087', 'NGC1365', 'NGC1385', 'NGC1433', 'NGC1566', 'NGC1672', 'NGC2835', 'NGC4254', 'NGC4303', 'NGC4321', 'NGC5068', 'NGC7496']

y = 0.015 # Kobayashi+20

for diag in ['Scal']:#, "N2S2Ha", 'Scal']:
	# Set up containers
	phi_lower_inner  = []
	phi_median_inner = []
	phi_upper_inner  = []
	
	phi_lower_outer  = []
	phi_median_outer = []
	phi_upper_outer  = []
	
	eps_lower_inner  = []
	eps_median_inner = []
	eps_upper_inner  = []
	
	eps_lower_outer  = []
	eps_median_outer = []
	eps_upper_outer  = []
	for gal_ID in split_gals :
		# Read in results
		inner_local_results  = pd.read_pickle('../Results/emcee_exp_fits/derived_params/{0}_{1}_inner.pkl'.format(gal_ID, diag))
		outer_local_results  = pd.read_pickle('../Results/emcee_exp_fits/derived_params/{0}_{1}_outer.pkl'.format(gal_ID, diag))
		# load and save phis
		phi_lower_inner.append( inner_local_results['phi'][16] * 1000)
		phi_median_inner.append(inner_local_results['phi'][50] * 1000)
		phi_upper_inner.append( inner_local_results['phi'][84] * 1000)
		
		phi_lower_outer.append( outer_local_results['phi'][16] * 1000)
		phi_median_outer.append(outer_local_results['phi'][50] * 1000)
		phi_upper_outer.append( outer_local_results['phi'][84] * 1000)
		# load and save efficiencies
		eps_lower_inner.append( inner_local_results['delta_Z'][16] / y)
		eps_median_inner.append(inner_local_results['delta_Z'][50] / y)
		eps_upper_inner.append( inner_local_results['delta_Z'][84] / y)
		
		eps_lower_outer.append( outer_local_results['delta_Z'][16] / y)
		eps_median_outer.append(outer_local_results['delta_Z'][50] / y)
		eps_upper_outer.append( outer_local_results['delta_Z'][84] / y)
		
	# Compute asymmetric error bars
	errors_phi_inner = np.array([np.array(phi_median_inner) - np.array(phi_lower_inner), np.array(phi_upper_inner) - np.array(phi_median_inner)])
	errors_phi_outer = np.array([np.array(phi_median_outer) - np.array(phi_lower_outer), np.array(phi_upper_outer) - np.array(phi_median_outer)])
	
	# Plot for Phi
	fig, axes = plt.subplots( figsize=(5,5))
	phi_min = np.min((np.min(phi_lower_outer), np.min(phi_lower_inner)))
	phi_max = np.max((np.max(phi_upper_outer), np.max(phi_upper_inner)))
	x= np.linspace(phi_min, phi_max)
	
	plt.plot(x,x, alpha=0.5, color='black', linestyle='--')
	plt.errorbar(phi_median_inner, phi_median_outer, xerr=errors_phi_inner, yerr=errors_phi_outer, fmt='o', color='red')
	
	plt.xlim(phi_min, phi_max)
	plt.ylim(phi_min, phi_max)
	plt.xlabel("$\phi$ (inner disc)")
	plt.ylabel("$\phi$ (outer disc)")
	
	plt.tight_layout()
	fig.savefig(plot_path + 'phi_inner_v_outer_{0}.png'.format(diag), dpi=100)

	# Compute asymmetric error bars
	errors_eps_inner = np.array([np.array(eps_median_inner) - np.array(eps_lower_inner), np.array(eps_upper_inner) - np.array(eps_median_inner)])
	errors_eps_outer = np.array([np.array(eps_median_outer) - np.array(eps_lower_outer), np.array(eps_upper_outer) - np.array(eps_median_outer)])
	
	# Plot for eps
	fig, axes = plt.subplots(figsize=(5,5))
	eps_min = np.min((np.min(eps_lower_outer), np.min(eps_lower_inner)))
	eps_max = np.max((np.max(eps_upper_outer), np.max(eps_upper_inner)))
	x= np.linspace(eps_min, eps_max)
	
	plt.plot(x,x, alpha=0.5, color='black', linestyle='--')
	plt.errorbar(eps_median_inner, eps_median_outer, xerr=errors_eps_inner, yerr=errors_eps_outer, fmt='o', color='red')
	
	plt.xlim(eps_min, eps_max)
	plt.ylim(eps_min, eps_max)
	plt.xlabel("$\epsilon$ (inner disc)")
	plt.ylabel("$\epsilon$ (outer disc)")
	
	plt.tight_layout()
	fig.savefig(plot_path + 'eps_inner_v_outer_{0}.png'.format(diag), dpi=100)