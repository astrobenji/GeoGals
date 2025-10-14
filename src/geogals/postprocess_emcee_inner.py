'''
postprocess_emcee_inner.py

Place to put functions to:

- read an emcee backend
- print chain plots
- ask me for the burn in 
- using that, create and save corner plots
- and also tell me the ac time for everything.

Once I have built these I will only need to save backends for everything

Last updated: Apr 04, 2025
'''

import numpy as np 
import matplotlib.pyplot as plt
import emcee
import corner
import pandas as pd
import GeoGals as gg

# Asplund+21
solar_Z_obs	   = 8.69	# log([O/H]) + 12
solar_Z_theory = 0.0139 # mass fraction of heavy elements

def read_emcee_result(gal_ID, diag):
	'''
	Give me the backend for the emcee file 
	showing the fits result for a given galaxy,
	analysed with a given diagnostic.
	'''
	return emcee.backends.HDFBackend('../Results/emcee_backends/subsamp_split/{0}_{1}_inner.hdf5'.format(gal_ID, diag), read_only=True)

def remove_low_f_ac_chains(chains, f_acc_threshold=0.2):
	diff_chains  = chains[1:,:,1] - chains[:-1,:,1]
	not_accepted = (diff_chains == 0)
	f_acc        = 1 - (np.sum(not_accepted, axis=0) / len(not_accepted))
	keep_rows    = (f_acc > f_acc_threshold)
	return chains[:,keep_rows]
	
def flatten(chains):
	'''
	Input: A M x N x 4 np array
	Output: A MN x 4 np array
	'''
	return chains.reshape(-1, 4)

plot_path	 = '../Plots/emcee_visual_checks/subsamp_split/inner_tests_'
RESOLUTION	 = 100 # dpi

labels = ['log_Var', 'phi_kpc', 'Z_char', 'grad_Z_per_kpc']
short_labels = ['logVar', 'phi', 'Zchar', 'gradZ']

split_gal_names = ['IC5332', 'NGC1087', 'NGC1365', 'NGC1385', 'NGC1433', 'NGC1566', 'NGC1672', 'NGC2835', 'NGC4254', 'NGC4303', 'NGC4321', 'NGC5068', 'NGC7496']

wonky_Scal_split_gals = ['NGC1087', 'NGC1365', 'NGC1566', 'NGC4254', 'NGC4303']

diags = ['N2S2Ha', 'O3N2', 'Scal']#'N2S2Ha', 'O3N2', 'Scal']# 'N2S2Ha', 'O3N2', 'Scal']

for gal_ID in wonky_Scal_split_gals:
	for diag in ['Scal']:
		# - read an emcee backend -- use one of these 2 lines while we are in testing.
		# Later we'll return to the nice TYPHOON-wrangling function
		# result =	emcee.backends.HDFBackend('../Data/emcee_backends/tests/{0}_{1}_f={2}_re.hdf5'.format(gal_ID, diag, f))
		# result =	emcee.backends.HDFBackend('../Data/emcee_backends/tests/{0}_{1}_f={2}_r25.hdf5'.format(gal_ID, diag, f))
		result = read_emcee_result(gal_ID, diag)
		chains = result.get_chain()

		#########################################
		#										#
		#		  Mode 1: Manual Burn-In		#
		#										#
		#########################################

		# To get burn-in, just look at the chain of params we directly sample over:
# 		fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
# 		for i in range(4):
# 			ax = axes[i]
# 			ax.plot(chains[:, :, i], "k", alpha=0.3)
# 			ax.set_xlim(0, len(chains))
# 			ax.set_ylabel(labels[i])
# 			ax.yaxis.set_label_coords(-0.1, 0.5)
# 			
# 		axes[-1].set_xlabel("step number")
# 		plt.show()
# 		burn_in = int(input("How long is the burn-in? "))
# 		plt.close(fig)

		#########################################
		#										#
		#		 Mode 2: 2AC time Burn-In		#
		#										#
		#########################################

		# better for automation/repetition
		burn_in = 120

		# - create and save chain plots

		# Make a figure with all 4 chains and save it
		chain_fig, axes = plt.subplots(4, figsize=(10, 8), sharex=True)
		axes[0].set_title("{0} {1} chains (burn-in: {2})".format(gal_ID, diag, burn_in))
		for i in range(4):
			ax = axes[i]
			ax.plot(chains[:, :, i], "k", alpha=0.3)
			ax.set_xlim(0, len(chains))
			ax.set_ylabel(labels[i])
			ax.yaxis.set_label_coords(-0.1, 0.5)
		axes[-1].set_xlabel("step number")
		# Save fig
		chain_fig.savefig(plot_path + '{0}_{1}_chains.png'.format(gal_ID, diag), dpi=RESOLUTION)
		plt.close()
		# - create and save corner plots

		samples      = result.get_chain(discard=burn_in)
		cut_samples  = remove_low_f_ac_chains(samples)
		flat_samples = flatten(cut_samples)

		flat_samples_w_4_params = flat_samples
		corner_fig = corner.corner(flat_samples_w_4_params, labels = ['$\log(\sigma^2)$', '$\phi$ (pc)', '$Z_{char}$', '$\\nabla Z$ (kpc$^{-1}$)' ])
		corner_fig.savefig(plot_path + '{0}_{1}_corner.png'.format(gal_ID, diag), dpi=RESOLUTION)
		plt.close()
		# Finally, save summary statistics for the models to a table 
		means = np.mean(flat_samples_w_4_params, axis=0)
		medians = np.median(flat_samples_w_4_params, axis=0)
		#print(medians)
		sd = np.std(flat_samples_w_4_params, axis=0)
		# I'm calling this "robust sd" -- but it's really half the distance from the 16th to 84th percentile
		percentile_16 = np.percentile(flat_samples_w_4_params, 16, axis=0)
		percentile_84 = np.percentile(flat_samples_w_4_params, 84, axis=0)
		percentile_sd = (percentile_84 - percentile_16)/2
		rho_matrix =  np.corrcoef(flat_samples_w_4_params, rowvar=False)
		data_dict = {
		'mean':means,
		'median':medians,
		'percentile_16': percentile_16,
		'percentile_84': percentile_84,
		'sd':sd,
		'percentile_sd': percentile_sd
		}	

		for ii, label in enumerate(short_labels):
			data_dict['corr_v_'+label] = rho_matrix[ii]

		df = pd.DataFrame(data_dict, index=labels)

		df.to_pickle('../Results/emcee_exp_fits/all_trained_params/{0}_{1}_inner.pkl'.format(gal_ID, diag))
		
		# Compute fluctuation size in units of physical metallicity
		logVar = flat_samples[:,0]
		phi	   = flat_samples[:,1]
		Z_char = flat_samples[:,2]
		grad_Z = flat_samples[:,3]
		# Need: Median Radius of particles in each split.
		Hii_df = gg.open_inner_Hii_df(gal_ID)
		meta   = gg.meta_getter(gal_ID)
		
		r_char	 = 0.4*meta['R25_kpc']
		
		wanted_spaxels = ~np.isnan(Hii_df['Z_'+diag]) & ~np.isinf(Hii_df['Z_'+diag])
		r	= gg.RA_DEC_to_radius(Hii_df['RA'][wanted_spaxels], Hii_df['DEC'][wanted_spaxels], meta)
		r_median = np.median(r)
		
		# Using this, find Z_local of each sample
		local_Z_obs = Z_char + grad_Z*(r_median - r_char)
		
		delta_Z_obs = np.power(10, 0.5*logVar)
		
		delta_Z_theory = solar_Z_theory*(
						 np.power(10, local_Z_obs + delta_Z_obs - solar_Z_obs) - 
						 np.power(10, local_Z_obs - solar_Z_obs)
						 )
		
		# Save percentiles for this value and phi
		phi_percentiles	  = []
		delZ_percentiles  = []
		percentile_levels = [2.5, 16, 50, 84, 97.5]
		for p in percentile_levels:
			phi_percentiles.append(np.percentile(phi, p))
			delZ_percentiles.append(np.percentile(delta_Z_theory, p))
			
		data_dict = {'phi': phi_percentiles, 'delta_Z': delZ_percentiles}
		df = pd.DataFrame(data_dict, index=percentile_levels)
		df.to_pickle('../Results/emcee_exp_fits/derived_params/{0}_{1}_inner.pkl'.format(gal_ID, diag))
		

# - and also tell me the ac time for everything.
# do this last as it loves to throw errors.
print("ac times: {0}, {1}, {2}, {3}".format(*result.get_autocorr_time(discard=burn_in)))