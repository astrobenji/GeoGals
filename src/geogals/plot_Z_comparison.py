import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import GeoGals as gg

plot_path = '../Plots/Results/Z_comparison/'

diags  = ['O3N2', 'N2S2Ha', 'Scal']
colors = ['blue', 'green', 'red']

#Sort galaxies by their mass
metadata   = gg.open_metadata()
gal_IDs    = [meta['Galaxy_ID'] for meta in metadata]
gal_masses = [meta['log_M_star'] for meta in metadata]
gal_IDs    = [x for _, x in sorted(zip(gal_masses, gal_IDs))]

# Setup X axes
x = np.arange(len(metadata)) 
jitter = 0.2
x_axes = [x - jitter, x, x + jitter]


# For Z char:
fig, ax = plt.subplots(figsize= (12, 6))
for ii, diag in enumerate(diags):
	# Setup containers
	Z_char_lower  = np.zeros(len(metadata))
	Z_char_median = np.zeros(len(metadata))
	Z_char_upper  = np.zeros(len(metadata))
	for jj, gal_ID in enumerate(gal_IDs):
		global_results = pd.read_pickle('../Results/emcee_exp_fits/all_trained_params/subsamp/{0}_{1}.pkl'.format(gal_ID, diag))
		
		Z_char_lower[jj]  = global_results['percentile_16']['Z_char']
		Z_char_median[jj] = global_results['median']['Z_char']
		Z_char_upper[jj]  = global_results['percentile_84']['Z_char']
	# Compute asymmetric error bars
	errors = [Z_char_median - Z_char_lower, Z_char_upper - Z_char_median]
	plt.errorbar(x_axes[ii], Z_char_median, yerr=errors, color=colors[ii], fmt='o', capsize=7, label = diag, linestyle='')



ax.set_xticks(x)
ax.set_xticklabels(gal_IDs, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
fig.savefig(plot_path + 'Z_char.png', dpi=100)