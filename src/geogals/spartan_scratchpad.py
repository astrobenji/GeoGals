# session commands

# enter
goto_spartan

# navigate to folder
cd Geostatistics/ANAL/Code/

# Get needed packages
module load Python/3.11.3
module load IPython
module load SciPy-bundle/2023.07-Python-3.11.3

ipython

# Load in data
import GeoGals as gg
import numpy as np

# gal_df = gg.open_line_df('NGC4321')

gal_IDs = ['IC5332', 'NGC0628', 'NGC1087', 'NGC1300', 'NGC1365', 'NGC1385', 'NGC1433', 'NGC1512', 'NGC1566', 'NGC1672', 'NGC2835', 'NGC3351', 'NGC3627', 'NGC4254', 'NGC4303', 'NGC4321', 'NGC4535', 'NGC5068', 'NGC7496']

diags = ['N2S2Ha', 'O3N2', 'Scal']

for gal in gal_IDs:
	for diag in diags:
		Hii_df = gg.open_Hii_df(gal)
		wanted_spaxels = ~np.isnan(Hii_df['Z_'+diag]) & ~np.isinf(Hii_df['Z_'+diag])
		print('{0} has {1} {2} spaxels'.format(gal, np.sum(wanted_spaxels), diag))

diags = ['O3N2', 'Scal', 'N2S2Ha']

for d in diags:
	print(d)
	inner_local_results  = pd.read_pickle('../Results/emcee_exp_fits/derived_params/{0}_{1}_inner.pkl'.format(gal_ID, d))
	outer_local_results  = pd.read_pickle('../Results/emcee_exp_fits/derived_params/{0}_{1}_outer.pkl'.format(gal_ID, d))
	print("Inner:")
	print(inner_local_results['phi'])
	print("local sfe:")
	print(inner_local_results['delta_Z'] / 0.015)
	print("outer:")
	print(outer_local_results['phi'])
	print("local sfe:")
	print(outer_local_results['delta_Z'] / 0.015)