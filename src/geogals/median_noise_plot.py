'''
Find median noise maps for all galaxies; is Scal high?
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import GeoGals as gg

diags  = ['O3N2', 'N2S2Ha', 'Scal']
colors = ['blue', 'green', 'red']

#Sort galaxies by their mass
metadata   = gg.open_metadata()
gal_IDs    = [meta['Galaxy_ID'] for meta in metadata]
gal_masses = [meta['log_M_star'] for meta in metadata]
gal_IDs    = [x for _, x in sorted(zip(gal_masses, gal_IDs))]

plot_path = '../Plots/Results/Z_comparison/'

# Setup X axes
x = np.arange(len(metadata)) 
jitter = 0.2
x_axes = [x - jitter, x, x + jitter]

e_Z = dict()

# For Z char:
fig, ax = plt.subplots(figsize= (12, 6))
for ii, diag in enumerate(diags):
	# Setup containers
	e_Z[diag] = np.zeros(len(metadata))
	for jj, gal_ID in enumerate(gal_IDs):
		Hii_df = gg.open_Hii_df(gal_ID)
		e_Z[diag][jj] = np.nanmedian(Hii_df['e_Z_'+diag])
	ax.scatter(x_axes[ii], e_Z[diag],  color=colors[ii], marker='o', label = diag)
	
ax.set_xticks(x)
ax.set_xticklabels(gal_IDs, rotation=45, ha='right')
ax.legend()
ax.set_ylabel("Uncertainty on Z (dex)")
plt.tight_layout()

fig.savefig(plot_path + 'e_Z.png', dpi=100)