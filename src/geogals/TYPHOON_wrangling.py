'''
TYPHOON_wrangling.py

Set of helper functions designed to help process TYPHOON data. 
Built out of phangs_phunctions.py

Created by: Benjamin Metha
Last updated: Jan 17, 2024
'''

import numpy as np
from astropy.io import fits
from sklearn.metrics.pairwise import euclidean_distances 
import pandas as pd 
from astropy.wcs import WCS
import astropy.units as u
from statsmodels.regression.linear_model import GLS
#from ltsfit.lts_linefit import lts_linefit
from scipy.linalg import cho_factor, cho_solve
from scipy.special import airy
from extinction import ccm89, apply
from Z_diags import *
import emcee

data_path = '../Data/'

# This folder contains some other TYPHOON galaxies at native res.
old_data_path = '../../krig_dig/Data/'

# BPT classification codes
STARBURST = 0
SEYFERT	  = 1
LINER	  = 2

ASEC_PER_RAD = 206265.0

class InputError(Exception):
	pass

#############################################
#											#	
#			  input/output					#
#											#
#############################################

def check_diag(diagnostic):
	'''
	Tells you whether a diagnostic is valid; throws an error if not.
	'''
	diagnostic = diagnostic.lower()
	ok_diags=['n2o2', 'o3n2', 'n2s2ha', 'o3s2_old','rs32', 'scal']
	if diagnostic not in ok_diags:
		raise ValueError("Error: "+ diagnostic +" is not a valid diagnostic.\n Choose from:"+str(ok_diags)[1:-1])
	return 0
	
def open_Z_df_topres_selection(gal_ID, res, diag):
	return pd.read_pickle(data_path + '/Bin_w_S2Kaplan_selection/{0}/{1}pix/Z_{2}.pkl'.format(gal_ID, res, diag))

# Deprecated
def open_metallicity_map(diagnostic='o3n2', gal_name='N5236', error=False):
	'''
	Opens metallicity map for a given galaxy, using a given metallicity diagnostic
	
	Parameters
	----------
	diagnostic: str
		Tells us which metallicity diagnostic to use.
		Options are ['n2ha', 'n2o2', 'n2s2', 'o3n2', 'r23'].
		Upper case values also accepted.
		Defaults to 'o3n2'
	
	gal_name: str
		Name of the galaxy. 
		Currently only N5236 is downloaded/has data, so it defaults to that.
		But it's a parameter you can change in the future.
		
	error: bool
		Do you want the error map for this diagnostic, too?

	Returns
	-------
	
	Z_map: np.array
	   Map for the metallicity of the galaxy at each location
	
	e_Z_map: np.array
		if error==True, I'll also give you the error map for this diagnostic.
	'''
	diagnostic = diagnostic.lower()
	check_diag(diagnostic)
	Z_hdu = fits.open(data_path+'{0}/{0}_full_41pc_MetIon_{1}.fits'.format(gal_name, diagnostic))
	if error:
		e_Z_hdu = fits.open(data_path+'{0}/{0}_full_41pc_meterror_{1}.fits'.format(gal_name, diagnostic))
		return Z_hdu[0].data, e_Z_hdu[0].data
	# Otherwise just give me the metallicity map, no error
	return Z_hdu[0].data
	
def open_uniformly_selected_line_df(gal_ID, res):
	'''
	Get the galaxy line dfs that were computed by binning over the spaxels that
	are selected to be Hii dominated at native resolution.
	
	Parameters:
	----------
	
	gal_ID: str 
		Galaxy identification string ('N' + 4 numbers)
	
	res: int
		Resolution in pixels
	
	'''
	hdu_list = fits.open(data_path + 'Bin_w_S2Kaplan_selection/{0}/clean_lines_f={1}.fits'.format(gal_ID, res))
	return hdu_list

def open_uniformly_selected_Hii_df(gal_ID, res):
	'''
	Get a table containing the metallicity and position of all spaxels, 
	selected using a classification scheme performed at native resolution.
	
	Parameters:
	----------
	
	gal_ID: str 
		Galaxy identification string ('N' + 4 numbers)
	
	res: int
		Resolution in pixels
	
	'''
	return pd.read_pickle(data_path + 'Bin_w_S2Kaplan_selection/{0}/Hii_df_f={1}.pkl'.format(gal_ID, res))

def open_Hii_map(gal_name='N5236'):
	"""
	Parameters
	----------
	gal_name: str
		Name of the galaxy. 
		Currently only N5236 is downloaded/has data, so it defaults to that.
		But it's a parameter you can change in the future.
		
	Returns
	-------
	
	Hii_map: np.array
	   Value of 1 if we're in a Hii region, 
	   0 if we are DIG dominated,
	   and NaN if there is no data.
	"""
	Hii_hdu = fits.open(data_path+gal_name+'/Hii_region.fits')
	return Hii_hdu[0].data

# Deprecated
def M83_metadata():
	'''Yes I should save this as a data object I can read. 
	No, I won't do that.
	'''
	M83_meta = {'D': 4.66,		 #Mpc (Tully et al. 2013)
				'i': 15.3,		 #Lauberts & Valentijn (1989)
			   'PA': 54.0,		 #Lauberts & Valentijn (1989)
			   'RA':204.2539583, #Dıaz et al. (2006)
			  'DEC':-29.8654167	 #Dıaz et al. (2006)
				   }
	return M83_meta
	
def open_metadata():
	'''
	Open the metadata file; saves from re-writing this code in every script,
	makes scripts more readable, and future-proofs code if I need to update
	or move the metadata file.
	'''
	meta_df = pd.read_csv(data_path+'metadata.csv')
	metadata = meta_df.to_dict(orient='records')
	return metadata
	
def meta_getter(gal_ID):
	metadata = open_metadata()
	meta   = [x for x in metadata if str(gal_ID) in x['Gal_ID']][0]
	return meta

# Deprecated
def open_Hii_table(res):
	return pd.read_pickle(data_path+'Handmade/{0}pix/Hii_table.pkl'.format(res))
	
def open_IC_data(gal_ID, diag, f):
	return np.load(data_path+'new_gradZ_ICs/{0}_{1}_f={2}.npy'.format(gal_ID, diag, f))
	
def open_line_df(gal_ID):
	'''
	Parameters:
	----------
	gal_ID: str 
		Galaxy identification string
	
	'''
	# Change 'NGC_'... to N...
	if gal_ID[:4] == 'NGC_':
		gal_ID = 'N'+gal_ID[4:]
	hdu_list = fits.open(data_path + 'WCS_line_maps/{0}_lowres_cal_1_comp_WCS_lines.fits'.format(gal_ID))
	# Trim the last 6 files because we just want line maps, and we don't care about chi squared, V disp, etc.
	return hdu_list[:26]
	
# Deprecated
def open_other_galaxy(gal_ID):
	'''
	Open a different galaxy to M83 from TYPHOON at native resolution, for cross checks.
	'''
	hdu_list = fits.open(old_data_path + 'WCS_line_maps/N{0:04d}_lowres_cal_1_comp_WCS.fits'.format(gal_ID))
	# Trim the last 6 files because we just want line maps, and we don't care about chi squared, V disp, etc.
	return hdu_list[:26]
	

def open_Z_model(gal_ID, diag):
	'''
	Parameters:
	----------
	
	gal_ID: str 
		Galaxy identification string ('N' + 4 numbers)
		
	diag: str, optional
		If supplied: what DIAG do you want?
		Must be supplied iff method == emcee
		
	method: 
	'''
	return np.load(data_path +'Handmade/gradZ/{0}_{1}.npy'.format(gal_ID, diag))

def read_emcee_CV_model(gal_ID, diag, res):
	return pd.read_pickle('../Data/Handmade/Models/emcee_exp_CV/{0}_{1}_f={2}.pkl'.format(gal_ID, diag, res))

def read_emcee_CV_subsamp_model(gal_ID, diag, n_dp):
	return pd.read_pickle('../Data/Handmade/Models/emcee_exp_CV/subsamp/{0}_{1}_n={2}.pkl'.format(gal_ID, diag, n_dp))

# Deprecated
def open_other_Z_grad_data(gal_ID, by='combo'):
	return pd.read_pickle(data_path +'Handmade/Other_Galaxies/Z_model_matern_by_{1}_N{0:04d}.pkl'.format(gal_ID, by))

##### TO TEST
def open_Hii_df(gal_ID, by='combo'):
	return pd.read_pickle(data_path +'Handmade/Full_galaxy_Z_dfs/Hii_by_{0}_{1}.pkl'.format(by, gal_ID))
			
# Deprecated
def open_other_Hii_df(gal_ID, by='combo'):
	return pd.read_pickle(data_path +'Handmade/Other_Galaxies/Hii_by_{0}_N{1:04d}.pkl'.format(by, gal_ID))
	
#def open_svg_model_data(gal_name='N5236'):
# deprecated
#	 return pd.read_pickle(data_path + 'Handmade/{0}/powerlaw_cutoff_model_fits_v1.pkl'.format(gal_name))

# Deprecated
def open_DIG_df(res, by='S2_Kaplan16'):
	return pd.read_pickle('../Data/Handmade/{0}pix/DIG_by_{1}.pkl'.format(res, by))

# Deprecated
def open_DIG_predictions_df(gal_name='N5236', by='S2_Kaplan16'):
	return pd.read_pickle('../Data/Handmade/{0}/DIG_by_{1}+predictions.pkl'.format(gal_name, by))

# Deprecated
def open_Hii_kriging_predictions(res, diag, by='combo'):
	return pd.read_pickle('../Outfiles/CV/{0}pix/{2}_{1}_Hii_CV_results.pkl'.format(res, diag, by))

# Deprecated
def open_Hii_kriging_predictions_vs_n_dp(n_dp, diag, by='combo', res=1):
	return pd.read_pickle('../Outfiles/CV/{0}pix/vs_N_DP/{2}_{1}_Hii_CV_results_{3}_dp.pkl'.format(res, diag, by, n_dp))

def open_subsample_indices(diag, by='combo', res=1):
	if by=='combo':
		with open('../Data/Handmade/vs_N_dp/spaxels_{1}pix/subsample_IDs_{0}.pkl'.format(diag, res), 'rb') as f:
			ID_dict = pickle.load(f)
		return ID_dict	
	elif by=='PHX':
		with open('../Data/Handmade/vs_N_dp/regions_{1}pix/subsample_IDs_{0}.pkl'.format(diag, res), 'rb') as f:
			ID_dict = pickle.load(f)
		return ID_dict	
	else:
		raise NotImplementedError('The only supported DIG rules are "combo" and "PHX".')

def open_Z_model_subsample(gal_ID, diag, n_dp):
	return pd.read_pickle('../Data/Handmade/Models/emcee_exp_fits/subsamp/{0}_{1}_n={2}.pkl'.format(gal_ID, diag, n_dp))

def Re_kpc(meta):
	'''Takes in a galaxy's metadata. Returns a conversion factor that converts
	kpc to Re.'''
	Re_radians = meta['Re (arcmin)']*60/ASEC_PER_RAD
	Re_kpc = Re_radians*1000*meta['D']
	return Re_kpc
	
def read_emcee_result(gal_ID, diag, f, special=None):
	'''
	Give me the backend for the emcee file 
	showing the fits result for a given galaxy,
	analysed with a given diagnostic,
	at a given resolution
	
	OPTIONAL str "special"
	can be "PHX" or "subsample"
	to get results from those folders
	'''
	if special is None:
		return emcee.backends.HDFBackend(data_path + 'emcee_backends/Results_main/{0}_{1}_f={2}.hdf5'.format(gal_ID, diag, f), read_only=True)
	elif special == 'subsamp':
		return emcee.backends.HDFBackend(data_path + 'emcee_backends/Results_{3}/{0}_{1}_n={2}.hdf5'.format(gal_ID, diag, f, special), read_only=True)
	else:
		# Get results from a special results folder
		return emcee.backends.HDFBackend(data_path + 'emcee_backends/Results_{3}/{0}_{1}_f={2}.hdf5'.format(gal_ID, diag, f, special), read_only=True)
		
def read_emcee_CV_result(gal_ID, diag, f, t, special=None):
	'''
	Give me the backend for the emcee file 
	showing the fits result for a given galaxy,
	analysed with a given diagnostic,
	at a given resolution
	
	OPTIONAL str "special"
	can be "PHX" or "subsample"
	to get results from those folders
	'''
	if special is None:
		return emcee.backends.HDFBackend(data_path + 'emcee_backends/Results_CV/{0}_{1}_f={2}_t={3}.hdf5'.format(gal_ID, diag, f, t), read_only=True)
	elif special=='PHX':
		# Get results from a special results folder
		return emcee.backends.HDFBackend(data_path + 'emcee_backends/Results_CV_{4}/{0}_{1}_f={2}_t={3}.hdf5'.format(gal_ID, diag, f, t, special), read_only=True)
	elif special=='subsamp':
		# Get results from a special results folder
		return emcee.backends.HDFBackend(data_path + 'emcee_backends/Results_{4}_CV/{0}_{1}_f=1_n={2}_t={3}.hdf5'.format(gal_ID, diag, f, t, special), read_only=True)
	else:
		print("Error: {0} is not a special result that I have.".format(special))
		exit()
	
#############################################
#											#
#			  data wrangling				#
#											#
#############################################

def construct_WCS(gal_name='N5236'):
	'''
	Construct the WCS from that thing you were given.
	'''
	calibration_file = fits.open(data_path+'WCS_line_maps/{0}_lowres_cal_1_comp_WCS.fits'.format(gal_name))
	w = WCS(calibration_file[0].header)
	return w

def make_RA_DEC(x_pix_list, y_pix_list, world):
	"""Converts a list of x and y pixels into lists of RA/DEC coordinates,
	given a WCS system."""
	if len(x_pix_list) != len(y_pix_list):
		raise InputError("Error in make_RA_DEC: lengths of x- and y- lists don't match!")
	n = len(x_pix_list)
	RA_list = np.empty(n)
	DEC_list = np.empty(n)
	for ii in range(n):
		coord = world.pixel_to_world(y_pix_list[ii], x_pix_list[ii])
		RA_list[ii]	 = coord.ra.value
		DEC_list[ii] = coord.dec.value
	return RA_list, DEC_list

def make_RA_DEC_grid(header):
	'''
	Given a header file, create a grid of RA//DEC for each pixel in that file.
	'''
	world = WCS(header)
	x = np.arange(header['NAXIS1'])
	y = np.arange(header['NAXIS2'])
	X, Y = np.meshgrid(x, y)
	RA_grid, DEC_grid = world.wcs_pix2world(X, Y, 0)
	return RA_grid, DEC_grid		
	
def vector_min(A, constant):
	'''
	A handy yet opaque trick I have used many times, turned into a function
	for readability.
	
	Takes in an array A and a constant. 
	
	Returns an array whose values are the same as A if A[i] < const,
	and const if A[i] > const.
	
	Basically a faster? version of [min(a, constant) for a in A]
	
	'''
	return A*(A<constant) + constant*(A>=constant)
	
def weighted_var(x, weights):
	'''
	Compute weighted variance using the formula from:
	https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
	with bias correction from:
	https://stats.stackexchange.com/questions/47325/bias-correction-in-weighted-variance?newreg=c21800bb15b4412995bee8ff31935596 
	'''
	if len(x) < 2:
		return np.nan
	wt_mean = np.average(x, weights=weights)
	wt_var	= np.average((x-wt_mean)**2, weights=weights)
	lambdas = weights/np.sum(weights)
	unbiasing_factor = 1 - np.sum(lambdas**2)
	return wt_var/unbiasing_factor

def unpack_and_trim(Hii_df, diag, dtype='f4'):
	'''
	Trim nans. Return Z, e_Z, RA, and DEC associated with the finite, non-nan
	values for a specified diagnostic.
	'''
	check_diag(diag)
	wanted_spaxels = ~np.isnan(Hii_df['Z_'+diag]) & ~np.isinf(Hii_df['Z_'+diag])
	Z	= np.array(Hii_df['Z_'+diag][wanted_spaxels],dtype=dtype)
	e_Z = np.array(Hii_df['e_Z_'+diag][wanted_spaxels],dtype=dtype)
	RA	= np.array(Hii_df['RA'][wanted_spaxels],dtype=dtype)
	DEC = np.array(Hii_df['DEC'][wanted_spaxels],dtype=dtype)
	return RA, DEC, Z, e_Z

def unpack_Z_and_Z_preds(Hii_df, diag, dtype='f4'):
	'''
	Trim nans. Return Z, e_Z, pred_Z, and e_pred_Z associated with the 
	non-nan values for a specified diagnostic.
	'''
	check_diag(diag)
	wanted_spaxels = ~np.isnan(Hii_df['Z_'+diag]) & ~np.isinf(Hii_df['Z_'+diag])
	Z	= np.array(Hii_df['Z_'+diag][wanted_spaxels],dtype=dtype)
	e_Z = np.array(Hii_df['e_Z_'+diag][wanted_spaxels],dtype=dtype)
	pred_Z	= np.array(Hii_df['pred_Z_'+diag][wanted_spaxels],dtype=dtype)
	e_pred_Z = np.array(Hii_df['e_pred_Z_'+diag][wanted_spaxels],dtype=dtype)
	return Z, e_Z, pred_Z, e_pred_Z

def count_non_nans(A):
	'''Handy debugginator'''
	return np.sum(~np.isnan(A))

#############################################
#											#
#	  processing emission line data into	#
#		 metallicity and DIG/Hii maps		#
#											#
#############################################

def SN_cut(line_df, threshold=3):
	'''
	Replace all spaxels with SN<3 in a certain line with NANs.
	
	Parameters
	----------
	
	lines_df: hdu list
		A big guy containing all the different emission line data reduced
		from TYPHOON data cubes
		
	threshold: float
		At what S/N do we cut a line? (Defaulted to 3)
		
	Returns
	-------
	lines_df: hdu list
		The same hdu list, but with lines where S/N < threshold
		replaced with np.nan
	'''
	n_lines = int(len(line_df)/2)
	x_max, y_max = line_df[0].data.shape
	for l in range(n_lines):
		signal = line_df[2*l].data
		noise  = line_df[2*l+1].data
		too_low = signal <= threshold*noise
		# Skip this step if the line is O2 - there's an error in the estimates
		# in this, so S/N cuts ought not be trusted. (Battisti, private 
		# communication, Oct 1 2021)
		# In this case, just exclude spaxels with S/N=0.
		if line_df[2*l].header['EXTNAME'] == 'OII3726' or line_df[2*l].header['EXTNAME'] == 'OII3729':
			too_low = (signal <= 0)
		for ii in range(x_max):
			for jj in range(y_max):
				# replace low signals/no signals with NANs.
				if too_low[ii,jj]:
					signal[ii,jj] = np.nan
					noise[ii,jj]  = np.nan
	
	return line_df
	
def SN_cut_O2(line_df, threshold=1):
	'''
	Separate program for the O2 line because it's special.
	'''
	n_lines = int(len(line_df)/2)
	x_max, y_max = line_df[0].data.shape
	for l in range(n_lines):
		signal = line_df[2*l].data
		noise  = line_df[2*l+1].data
		if line_df[2*l].header['EXTNAME'] == 'OII3726' or line_df[2*l].header['EXTNAME'] == 'OII3729':
			too_low = signal <= threshold*noise
		else:
			too_low = (signal <= 0)
		for ii in range(x_max):
			for jj in range(y_max):
				# replace low signals/no signals with NANs.
				if too_low[ii,jj]:
					signal[ii,jj] = np.nan
					noise[ii,jj]  = np.nan
	
	return line_df
	
def extinction_correction(line_df, wavelengths, R_V=3.1):
	'''
	Parameters
	----------
	
	lines_df: hdu list
		A big guy containing all the different emission line data reduced
		from TYPHOON data cubes
		
	wavelengths: np.array
		Wavelength of each of the 8 lines in this data cube, in Angstroms.
		
	R_V: float
		The free parameter in ccm89 extinction law. Set (kept) at 3.1.
	
	Returns
	-------
	
	corrected_lines_df: hdu list
		Corrections for all lines using the calibration of ccm89.
	'''
	line_IDs = [line_df[x].header['EXTNAME'] for x in range(len(line_df))] # the who's who of line data
	Ha_map = line_df[line_IDs.index('HALPHA')].data
	Hb_map = line_df[line_IDs.index('HBETA')].data
	# To convert balmer decrement to extinction, need these...
	HA_EXT =  ccm89(np.array([6562.8]), 1.0, R_V)[0]
	HB_EXT =  ccm89(np.array([4861.3]), 1.0, R_V)[0]
	Ha_Hb_ratio	 = Ha_map/Hb_map
	balmer_decrement = 2.5*np.log10(Ha_Hb_ratio / 2.86)
	A_V = balmer_decrement/(HB_EXT - HA_EXT) 
	A_V_positive = A_V * (A_V > 0) # sets negatives to zero
	
	# Use this to correct obs and error for each wavelength
	for l in range(len(wavelengths)):
		extinction_at_wav = ccm89(wavelengths[l:l+1], 1, R_V)[0]
		extinction_map = extinction_at_wav*A_V_positive
		# correct signal and noise
		line_df[2*l].data	 = line_df[2*l].data * 10**(0.4 * extinction_map)
		line_df[2*l+1].data	 = line_df[2*l+1].data * 10**(0.4 * extinction_map)
	
	return line_df

def determine_DIG_S2_Kaplan16(line_df, n_spaxels=100, max_prop=0.05):
	'''
	Assuming that:
	1. The Sii/Ha line ratio is significantly different for DIG/Hii regions;
	2. The intrinsic distributions of Sii/Ha are (infinitely) narrow for purely Hii/DIG regions
	
	Compute the fraction of Ha-light originating from Hii regions for each spaxel
	(C_Hii), using the formalism of Kaplan+2016:
	https://ui.adsabs.harvard.edu/abs/2016MNRAS.462.1642K
	
	The formula:
	
	[Sii/Ha] = C_Hii [Sii/Ha]_Hii + C_DIG [Sii/Ha]_DIG
	
	Solvable when you use:
	
	C_Hii + C_DIG = 1
	[Sii/Ha]_Hii = the median [Sii/Ha] of the brightest spaxels
	[Sii/Ha]_DIG = the median [Sii/Ha] of the faintest spaxels
	
	Parameters
	----------
	
	lines_df: hdu list
		A big guy containing all the different emission line data reduced
		from TYPHOON data cubes
		
	n_spaxels: int
		A hyperparameter. Set to be 100 fiducually, it sets the number of spaxels
		used to compute intrinsic Hii or DIG Sii/Ha values.
		
	max_prop: float
		only use up to [max_prop] of spaxels to compute the intrinsic Hii or DIG 
		Sii/Ha values.
		
	Returns
	-------
	
	C_Hii: np. array
		Fraction of Ha light emanating from Hii regions, for each spaxel.
		
	Does not return errors associated with this calculation.
	'''
	# open wanted line data/get wanted line ratios
	line_IDs = [line_df[x].header['EXTNAME'] for x in range(len(line_df))]
	Ha = line_df[line_IDs.index('HALPHA')].data
	S2Ha = (line_df[line_IDs.index('SII6716')].data+line_df[line_IDs.index('SII6731')].data)/line_df[line_IDs.index('HALPHA')].data	 
	# only consider spaxels where both of these measures exceed the S/N theshold
	useful_spaxels = ~np.isnan(S2Ha)
	n_useful_spaxels = np.sum(useful_spaxels)
	useful_Ha = Ha[useful_spaxels]
	useful_S2Ha = S2Ha[useful_spaxels]
	n_per_group = np.min((n_spaxels, int(max_prop*n_useful_spaxels)))
	if n_per_group < n_spaxels:
		print("Computing median Sii/Ha for DIG/Hii regions using {0} of spaxels (5%) per group...".format(n_per_group))
	brightness_order = np.argsort(useful_Ha)
	# Assume the least bright spaxels are pure DIG
	# Compute their median S2Ha
	DIG_S2Ha = np.median(useful_S2Ha[brightness_order][:n_per_group])
	# Do the same for Hii
	Hii_S2Ha = np.median(useful_S2Ha[brightness_order][-1*n_per_group:])
	# Using these values, convert S2Ha to C_Hii
	C_Hii = (S2Ha - DIG_S2Ha)/(Hii_S2Ha - DIG_S2Ha)
	return C_Hii
	
def determine_DIG_SII_Kaplan_M83_hi_res(line_df):
	'''
	This time we get the typical Sii/Ha ratio from the lowest res map.
	ONLY CALIBRATED FOR M83; DON'T USE ON OTHER GALAXIES
	'''
	line_IDs = [line_df[x].header['EXTNAME'] for x in range(len(line_df))]
	S2Ha = np.log10( (line_df[line_IDs.index('SII6716')].data+line_df[line_IDs.index('SII6731')].data)/line_df[line_IDs.index('HALPHA')].data)
	# Two precomputed values from the high res map
	native_res_DIG_S2Ha = -0.07077161
	native_res_Hii_S2Ha = -0.67985165
	# Using these values, convert S2Ha to C_Hii
	return (S2Ha - native_res_DIG_S2Ha)/(native_res_Hii_S2Ha - native_res_DIG_S2Ha)

def determine_DIG_Ha_Zhang17(Ha_map, meta, cut=39):
	'''
	Determine whether a spaxel is Hii/DIG dominated, by applying a surface
	brightness cut in Ha.
	
	
	Formula: dig if log_10( SB_Ha ) < 39
	
	where SB_Ha is in units of erg s−1 kpc−2
	
	Parameters
	----------
	
	Ha_map: hdu
		the intensity of the Ha line data for this galaxy
	
	meta: dict-like object including:
	 D - distance to an object in Mpc.
	 i - inclination in degrees
	 
	cut: dfaults to 39 as recommended by Zhang.
		 But we leave the opportunity to remain creative open.
	
	Returns
	-------
	
	DIG_map: np array
		0 if a spaxel is DIG dominated
		1 if it's a Hii region
		nan if SN too low to decide.
	'''
	# Finagle out the deprojected area (units: kpc)
	world = WCS(Ha_map.header)
	pix_solid_angle = world.proj_plane_pixel_area().to(u.steradian).value
	plane_area	= pix_solid_angle * (meta['D']*1000)**2 # in kpc^2
	i  = np.radians(meta['i'])
	deproj_area = plane_area / np.cos(i)
	# convert flux (units:1e-17 erg/s/cm2) to Luminosity (units: erg/s)
	log_Ha_luminosity_map = np.log10(Ha_map.data) + np.log10(4*np.pi*meta['D']**2 * 9.5234) + 31
	# convert to surface brightnesss (units: erg/s/kpc^2)
	log_Ha_SB_map = log_Ha_luminosity_map - np.log10(deproj_area)
	is_Hii = log_Ha_SB_map > cut
#	is_nan = np.isnan(log_Ha_SB_map)
#	DIG_map = np.zeros(log_Ha_SB_map.shape)+ 1.0*is_Hii
#	X_MAX, Y_MAX = log_Ha_SB_map.shape
#	for ii in range(X_MAX):
#		for jj in range(Y_MAX):
#			if is_nan[ii,jj]:
#				DIG_map[ii,jj] = np.nan
	return is_Hii

def classify_S2_BPT(line_df):
	'''
	For each spaxel
	specify whether it is SEYFERT, LINER, or SF
	using the diagnostics of Kewley+01 and Kewley+06
	and the S2-BPT diagram.
	
	Parameters
	----------
	
	lines_df: hdu list
		A big guy containing all the different emission line data reduced
		from TYPHOON data cubes
		
	Returns
	-------
	
	S2_BPT_classification: np array
		True if in a Hii region
		False if not
		For all spaxels
	'''
	line_IDs = [line_df[x].header['EXTNAME'] for x in range(len(line_df))]
	O3Hb = np.log10( line_df[line_IDs.index('OIII5007')].data /	 line_df[line_IDs.index('HBETA')].data )
	S2Ha = np.log10( (line_df[line_IDs.index('SII6716')].data+line_df[line_IDs.index('SII6731')].data)/line_df[line_IDs.index('HALPHA')].data	 )
	is_starburst = O3Hb < ( 0.72/(S2Ha-0.32) + 1.3 )
	return is_starburst & (S2Ha < 0.32)

def classify_N2_BPT(line_df, rule="Kauffmann03"):
	'''
	For each spaxel
	specify whether it is LINER or SF
	using the diagnostic of Kewley+01
	and the N2-BPT diagram.
	
	Parameters
	----------
	
	lines_df: hdu list
		A big guy containing all the different emission line data reduced
		from TYPHOON data cubes
		
	Returns
	-------
	N2_BPT_classification: np array
		True if in a Hii region
		False if not
		For all spaxels
	'''
	line_IDs = [line_df[x].header['EXTNAME'] for x in range(len(line_df))]
	O3Hb = np.log10( line_df[line_IDs.index('OIII5007')].data/line_df[line_IDs.index('HBETA')].data )
	N2Ha = np.log10( line_df[line_IDs.index('NII6583')].data/line_df[line_IDs.index('HALPHA')].data	   )
	if rule=='Kewley01':
		is_starburst = O3Hb < 0.61/(N2Ha-0.47) + 1.19
		is_LINER	 = O3Hb >= 0.61/(N2Ha-0.47) + 1.19 # otherwise it's a NAN
	elif rule=='Kauffmann03':
		is_starburst = (O3Hb < 0.61/(N2Ha-0.05) + 1.3) 
		is_LINER	 = (O3Hb > 0.61/(N2Ha-0.05) + 1.3)
	else:
		print("Error: classsify_N2_BPT only works when 'rule' is either 'Kewley01' or 'Kauffmann03'.")
		exit(1)
		return None
	return is_starburst & (N2Ha < 0.05)
	
def extract_lines_for_spherical_HII_regions(line_df, Hii_table, n_lines = 13):
	'''
	Parameters
	----------
	line_df: hdu_list with many hdus.
		Half correspond to line fluxes in each spaxel
		Half correspond to errors associated with each line flux
		Names of each line are stored in 'EXTNAME' in header
		
	HII_table: pandas df with three columns:
		X: x-pixel location of spaxel
		Y: y-pixel location of spaxel
		R: radius of spaxel, in pixels.
		
	n_lines: int, default
		Number of different lines in this datacube.
	
	Returns
	-------
	Hii_line_ratios: pandas df with columns:
		ID: ID of Hii region
		RA: of Hii region centre
		DEC: of Hii reigon centre
		r_pc: radius of Hii reigon in pc
		*: Flux of emission line * (name taken from line_df) (added over all spax in region)
		*_err: Error of emission line * (added in quadrature from all spax in region)
		
	!!CAVEAT!! pyHIIextractor's Hii table may include Hii regions that overlap or
	share some pixels. We do not capture the correlations in such data. This is
	why I didn't want to do it like this... oey... 
	'''
	n_regions = len(Hii_table)
	# Initialise the output dictionary
	out = {'ID':np.arange(n_regions)}
	# Use WCS of line_df to convert X, Y of HII_table into RA, DEC
	world = WCS(line_df[0].header)
	RA_list, DEC_list = make_RA_DEC(Hii_table['X'], Hii_table['Y'], world)
	out['RA']	= RA_list
	out['DEC']	= DEC_list
	## WARNING! MAGIC NUMBER HERE OF 4.66 Mpc will need to be changed for other gals
	d_pc		= 4.66e6 # distance in parsec
	pix_to_pc	= world.proj_plane_pixel_scales()[0].to(u.rad).value * d_pc
	out['r_pc'] = np.array(Hii_table['R']) * pix_to_pc
	# initialise line dicts
	for ll in range(n_lines):
		line_name			= line_df[2*ll].header['EXTNAME']
		out[line_name]		= np.empty(n_regions)
		out['e_'+line_name] = np.empty(n_regions)
		
	# find the closest Hii region to each pixel
	n_x, n_y = line_df[0].data.shape
	X_grid, Y_grid = np.meshgrid(np.arange(n_x), np.arange(n_y))
	X_long = np.concatenate(X_grid)
	Y_long = np.concatenate(Y_grid)
	pixel_vector = np.stack((Y_long, X_long)).T
	Hii_vector = np.stack((np.array(Hii_table['Y']), np.array(Hii_table['X']))).T
	
	pixel_to_Hii_dists = euclidean_distances(pixel_vector, Hii_vector)
	closest_Hii_to_each_pixel = np.argmin(pixel_to_Hii_dists , axis=1)
	closest_Hii_ID_map = closest_Hii_to_each_pixel.reshape(n_y,n_x).T
	
	for ii in range(n_regions):
		pix_pairs = find_pix_pairs(Hii_table['X'][ii], Hii_table['Y'][ii], Hii_table['R'][ii])
		for ll in range(n_lines):
			line_name  = line_df[2*ll].header['EXTNAME']
			line_index = 2*ll
			err_index  = 2*ll + 1
			# sum line, err data
			flux_in_region = 0
			err2_in_region	= 0 
			for x_pix, y_pix in pix_pairs:
				try:
					if closest_Hii_ID_map[x_pix,y_pix] != ii:
						continue
				except IndexError:
					continue
				spaxel_flux = line_df[line_index].data[x_pix, y_pix]
				spaxel_err2 = line_df[err_index].data[x_pix, y_pix]**2
				if not np.isnan(spaxel_flux):
					flux_in_region += spaxel_flux
					err2_in_region += spaxel_err2 
			# Save these to the lists
			out[line_name][ii] = flux_in_region
			out['e_'+line_name][ii] = np.sqrt(err2_in_region)
	
	out_df = pd.DataFrame(out) # Q: Why do I care about pandas vs dict?
	return out_df
	
# def Hii_regions_vs_subspaxel_Z(line_df, Hii_table, n_to_draw=5):
# 	'''
# 	For all Hii regions generate two dicts:
# 	
# 	1. for the regions, save Z and e_Z for 3 metallicity diagnostics
# 	2. for each spaxel in the region, save its Z and e_Z separately.
# 	
# 	Returns
# 	-------
# 	
# 	Hii_region_dicts: list of dicts of floats
# 	Hii_subspaxel_dicts: list of dicts of arrays.
# 	
# 	For use in plotting or subsampling or smth later
# 	'''
# 	n_lines	  = 13
# 	n_regions = len(Hii_table)
# 	# Initialise the output dictionaries
# 	Hii_region_dicts    = [dict() for ii in range(n_regions)]
# 	Hii_subspaxel_dicts = [dict() for ii in range(n_regions)]
# 	# Use WCS of line_df to convert X, Y of HII_table into RA, DEC
# 	world = WCS(line_df[0].header)
# 	RA_list, DEC_list = make_RA_DEC(Hii_table['X'], Hii_table['Y'], world)
# 	## WARNING! MAGIC NUMBER HERE OF 4.66 Mpc will need to be changed for other gals
# 	d_pc		= 4.66e6 # distance in parsec
# 	pix_to_pc	= world.proj_plane_pixel_scales()[0].to(u.rad).value * d_pc
# 	# initialise line dicts
# 	for ii in range(n_regions):
# 		for ll in range(n_lines):
# 			line_name			= line_df[2*ll].header['EXTNAME']
# 			Hii_region_dicts[ii][line_name]	= 0
# 			Hii_subspaxel_dicts[ii][line_name] = []
# 			Hii_region_dicts[ii]['e_'+line_name]= 0
# 			Hii_subspaxel_dicts[ii]['e_'+line_name] = []
# 	
# 	# find the closest Hii region to each pixel
# 	n_x, n_y = line_df[0].data.shape
# 	X_grid, Y_grid = np.meshgrid(np.arange(n_x), np.arange(n_y))
# 	X_long = np.concatenate(X_grid)
# 	Y_long = np.concatenate(Y_grid)
# 	pixel_vector = np.stack((Y_long, X_long)).T
# 	Hii_vector = np.stack((np.array(Hii_table['Y']), np.array(Hii_table['X']))).T
# 	
# 	pixel_to_Hii_dists = euclidean_distances(pixel_vector, Hii_vector)
# 	closest_Hii_to_each_pixel = np.argmin(pixel_to_Hii_dists , axis=1)
# 	closest_Hii_ID_map = closest_Hii_to_each_pixel.reshape(n_y,n_x).T
# 	
# 	for ii in range(n_regions):
# 		pix_pairs = find_pix_pairs(Hii_table['X'][ii], Hii_table['Y'][ii], Hii_table['R'][ii])
# 		for ll in range(n_lines):
# 			line_name  = line_df[2*ll].header['EXTNAME']
# 			line_index = 2*ll
# 			err_index  = 2*ll + 1
# 			# sum line, err data
# 			flux_in_region = 0
# 			err2_in_region	= 0 
# 			for x_pix, y_pix in pix_pairs:
# 				try:
# 					if closest_Hii_ID_map[x_pix,y_pix] != ii:
# 						continue
# 				except IndexError:
# 					continue
# 				spaxel_flux = line_df[line_index].data[x_pix, y_pix]
# 				spaxel_err2 = line_df[err_index].data[x_pix, y_pix]**2
# 				if not np.isnan(spaxel_flux):
# 					flux_in_region += spaxel_flux
# 					err2_in_region += spaxel_err2 
# 			# Save these to the lists
# 			out[line_name][ii] = flux_in_region
# 			out['e_'+line_name][ii] = np.sqrt(err2_in_region)
# 	
# 	return Hii_region_dicts, Hii_subspaxel_dicts
	
def add_BPT_diagnostics(Hii_table):
	'''
	Adds a column to a Hii region table that indicates whether or not each
	Hii region passes the Kewley+01/Kauffmann+03 test for being a star-forming
	region. For consistency checks.
	
	Parameters
	----------
	Hii_table: pandas df with columns:
		ID: ID of Hii region
		RA: of Hii region centre
		DEC: of Hii reigon centre
		r_pc: radius of Hii reigon in pc
		*: Flux of emission line * (name taken from line_df) (added over all spax in region)
		*_err: Error of emission line * (added in quadrature from all spax in region)
		
	Returns
	-------
	The same table but with two extra columns:
	
	Kauffmann+03_HII: bool
		1 if this spaxel lies in the BPT region consistent with being a Hii region
		0 otherwise
		
	Kewley+01_HII: analogous, but with the Kewley+01 diagnostic instead.
	'''
	# Get relevant line ratios CHANGE THESE FOR THE NEW DATA STRUCTS
	O3Hb = np.log10(  Hii_table['OIII5007'] /  Hii_table['HBETA'] )
	S2Ha = np.log10( (Hii_table['SII6716'] + Hii_table['SII6731']) / Hii_table['HALPHA'] )
	N2Ha = np.log10(  Hii_table['NII6583'] / Hii_table['HALPHA'] )
	
	# First, let's do Kewley+01 with the OIII / Hb - SII/Ha diagram
	S2_BPT_classification = (O3Hb < ( 0.72/(S2Ha-0.32) + 1.3 )) & (S2Ha < 0.32)
	Hii_table['Kewley+01_HII'] = S2_BPT_classification
	
	# Now we do Kauffmann+03 diagnostic on OIII / Hb - NII/Ha diagram
	N2_BPT_classification = (O3Hb < 0.61/(N2Ha-0.05) + 1.3) & (N2Ha < 0.05)
	Hii_table['Kauffmann+03_HII'] = N2_BPT_classification
	
	return Hii_table
			
def find_pix_pairs(x,y,R):
	'''
	Find all pairs of integer values that are less than R from (x,y)
	
	Parameters
	----------
	x, y: ints
		Location of centre of circle
	
	R: float
		Radius of circle (pixel)
		
	Returns
	-------
	pix_list: list of (x,y) pairs of pixels within the circle
	'''
	x_to_test = np.arange(int(x) - int(R), int(x)+int(R)+1)
	y_to_test = np.arange(int(y) - int(R), int(y)+int(R)+1)
	long_x_test, long_y_test = np.meshgrid(x_to_test, y_to_test)
	pairs_to_test = np.stack((long_x_test.flatten(), long_y_test.flatten())).T
	in_circle = (euclidean_distances(pairs_to_test, np.array([[x,y]])) <= R)[:,0]
	return pairs_to_test[in_circle]
	
				
#############################################
#											#
#			  spatial statistics			#
#											#
#############################################

def deprojected_distances(RA1, DEC1, RA2=None, DEC2=None, meta=dict()):
	'''
	Computes the deprojected distances between one set of RAs/DECs and
	another, for a known galaxy.
	
	Parameters
	----------
	
	RA1: float, list, or np array-like
		List of (first) RA values. Must be in degrees.
		
	DEC1: float, list, or np array-like
		List of (first) DEC values. Must be in degrees.
		
	RA2: float, list, or np array-like
		(Optional) second list of RA values. Must be in degrees.
		If no argument is provided, then the first list will be used again.
		
	DEC2: float, list, or np array-like
		(Optional) second list of DEC values. Must be in degrees.
		If no argument is provided, then the first list will be used again.	   
	
	meta: dict
		Metadata used to calculate the distances. Must contain:
		PA: float
			Principle Angle of the galaxy, degrees.
		i: float
			inclination of the galaxy along this principle axis, degrees.
		D: float
			Distance from this galaxy to Earth, Mpc.
		
	Returns
	-------
	dists: np array
		Array of distances between all RA, DEC pairs provided.
		Units: kpc.
	
	'''
	# Check parameters
	try:
		meta['PA'] 
	except KeyError:
		assert False, "Error: PA not defined for metadata"
	try:
		meta['i'] 
	except KeyError:
		assert False, "Error: i not defined for metadata"
	try:
		meta['D'] 
	except KeyError:
		assert False, "Error: D not defined for metadata"
	
	# If RA1 and DEC1 are arrays, they must have the same length.
	# If one of them is a float, they must both be floats.
	# You can't supply only one of RA2 and DEC2
	try:
		assert len(RA1) == len(DEC1), "Error: len of RA1 must match len of DEC1"
		RA1 = np.array(RA1)
		DEC1 = np.array(DEC1)
	except TypeError:
		assert type(RA1) == type(DEC1), "Error: type of RA1 must match type of DEC1"  
		# Then cast them to arrays
		RA1 = np.array([RA1])
		DEC1 = np.array([DEC1])
		
	if type(RA2) == type(None):
		RA2 = RA1
	if type(DEC2) == type(None):
		DEC2 = DEC1
	
	try:
		assert len(RA2) == len(DEC2), "Error: len of RA2 must match len of DEC2"
		RA2 = np.array(RA2)
		DEC2 = np.array(DEC2)
	except TypeError:
		assert type(RA2) == type(DEC2), "Error: type of RA2 must match type of DEC2" 
		RA2 = np.array([RA2])
		DEC2 = np.array([DEC2])
	
	# Now onto the maths
	PA = np.radians(meta['PA'])
	i  = np.radians(meta['i'])
	# 1: Rotate RA, DEC by PA to get y (major axis direction) and x (minor axis direction)
	x1 = RA1*np.cos(PA) - DEC1*np.sin(PA)
	y1 = DEC1*np.cos(PA) + RA1*np.sin(PA)
	x2 = RA2*np.cos(PA) - DEC2*np.sin(PA)
	y2 = DEC2*np.cos(PA) + RA2*np.sin(PA)
	# 2: Stretch x values to remove inclination effects
	long_x1 = x1 /np.cos(i)
	long_x2 = x2 /np.cos(i)
	# 3: Compute Euclidean Distances between x1,y1 and x2,y2 to get angular offsets (degrees).
	vec1 = np.stack((y1, long_x1)).T
	vec2 = np.stack((y2, long_x2)).T
	deg_dists = euclidean_distances(vec1, vec2)
	rad_dists = np.radians(deg_dists)
	# 4: Convert angular offsets to kpc distances using D, and the small-angle approximation.
	Mpc_dists = rad_dists * meta['D']
	kpc_dists = Mpc_dists * 1000
	
	return kpc_dists

def RA_DEC_to_radius(RA, DEC, meta):
	return deprojected_distances(RA, DEC, meta['RA'], meta['DEC'], meta).T[0]

def build_error_covariance_matrix(dist_matrix, e_Z, meta=dict(), seeing=0.6):
	'''
	Build the covariance matrix due to correlated error associated with the 
	measurement of emission lines.
	Assumes PSF of the telescope is a Gaussian.
	Uses value of 0.6 arcseconds for the seeing:
	https://ui.adsabs.harvard.edu/abs/1990ExA.....1..195P/abstract
	
	Parameters
	----------
	
	dist_matrix: (N,N) np.array
		Distances between all pairs of regions.
		
	e_Z: (N,) np.array
		Uncertainty in metallicity for each observation 
		
	meta: dict
		Metadata used to calculate the distances. Must contain:
		D: float
			Distance from this galaxy to Earth, Mpc.
	
	Returns
	-------
	cov_matrix: (N,N) np.array
		Covariance matrix for correlated observation errors.
	'''
	# Convert seeing of 0.6'' to kpc, using small angle approximation
	physical_seeing = seeing*meta['D']*1000/ASEC_PER_RAD
	# Convert seeing (FWHM) into a s.d.
	seeing_sd = physical_seeing / (2*np.sqrt(2*np.log(2))) # from	
	# Assume the telescope has a Gaussian PSF:
	correlation_matrix = np.exp(-0.5* (dist_matrix/seeing_sd)**2)
	sd_matrix  = np.diag(e_Z)
	cov_matrix = sd_matrix @ correlation_matrix @ sd_matrix
	return cov_matrix
	
def powerlaw_cutoff_cov_matrix(dist_matrix, A, beta, cutoff):
	norm_dist = dist_matrix/cutoff
	correlation_matrix = (1 - norm_dist**beta)*(norm_dist < 1)
	return A*correlation_matrix
	
def exp_corr_matrix(dist_matrix, phi):
	return np.exp(-1.0*dist_matrix/phi)
	
def matern_Kolmogorov_correlation(dist_matrix, phi):
	'''
	From Lee+Gammie 2021
	A matern function that has nu = 1/3
	In the small limit this corresponds to being dominated by 
	Kolmogorov turbulence.
	
	Using identity can be expressed in terms of an Airy function.
	functions.wolfram.com/Bessel-TypeFunctions/BesselK/introductions/Bessels/05/
	'''
	prefactor = 2.81667882 # = 2 * pi * 3^(1/6) / (Gamma function of 1/3) < put into Wolfram Alpha
	r_on_phi  = dist_matrix / phi
	z  = np.power( 3* r_on_phi / 2 , 2/3)
	return prefactor * airy(z)[0]
	
def exp_semivariogram(x, diag, Z_grad_data):
	norm_dist = x/Z_grad_data['phi'][diag]
	semivariance = 1 - np.exp(-1.0*norm_dist)
	A = 10.0**Z_grad_data['log_Var'][diag] 
	return A * semivariance

def powerlaw_cutoff_semivariogram(x, diag, Z_grad_data):
	norm_dist = x/Z_grad_data['phi'][diag]
	norm_dist_maxxed = norm_dist * (norm_dist<1) + 1.0*(norm_dist>1)
	A = 10.0**Z_grad_data['log_Var'][diag] 
	semivariance = norm_dist_maxxed**Z_grad_data['beta'][diag]
	return A * semivariance
	
def powerlaw_cutoff_semivariogram_w_nugget(x, diag, Z_grad_data):
	norm_dist = x/Z_grad_data['phi'][diag]
	norm_dist_maxxed = norm_dist * (norm_dist<1) + 1.0*(norm_dist>1)
	A = 10.0**Z_grad_data['log_Var'][diag] 
	semivariance = norm_dist_maxxed**Z_grad_data['beta'][diag]
	return A * semivariance + Z_grad_data['nugget'][diag] 
	
def powerlaw_cutoff_cov_matrix_w_nug(dist_matrix, A, beta, cutoff, nugget):
	n=dist_matrix.shape[0]
	norm_dist = dist_matrix/cutoff
	correlation_matrix = (1 - norm_dist**beta)*(norm_dist < 1)
	nug_matrix = np.diag(np.ones(n)*nugget)
	return A*correlation_matrix + nug_matrix

def compute_Z_grad(Hii_df, diag, meta, verbose=True):
	# Remove nans
	wanted_spaxels = ~np.isnan(Hii_df['Z_'+diag])
	# Find all Hii spaxels' distance from centre
	r_kpc  = RA_DEC_to_radius(Hii_df['RA'][wanted_spaxels], Hii_df['DEC'][wanted_spaxels], meta)
	# Find all Hii spaxels' distance from each other
	dist_matrix = deprojected_distances(Hii_df['RA'][wanted_spaxels], Hii_df['DEC'][wanted_spaxels], meta=meta)
	e_Z = build_error_covariance_matrix(dist_matrix, e_Z=Hii_df['e_Z_'+diag][wanted_spaxels], meta=meta)
	# Compute linear regression using GLS
	covariates = np.array([np.ones(len(r_kpc)), r_kpc]).T 
	Z_grad_model = GLS(Hii_df['Z_'+diag][wanted_spaxels], covariates, sigma=e_Z).fit()
	Z_centre, Z_grad = Z_grad_model.params
	e_Z_centre = np.sqrt(Z_grad_model.cov_params())['const']['const']
	e_Z_grad   = np.sqrt(Z_grad_model.cov_params())['x1']['x1']
	if verbose:
		print("# of spaxels:%d" % len(r_kpc))
		print(Z_grad_model.summary())
	return Z_centre, Z_grad, e_Z_centre, e_Z_grad
	
def compute_Z_grad_iteratively(Hii_df, diag, meta, powerlaw_parameters):
	'''
	Compute the metallicity gradient, using 1. the correlated errors in
	measurement (assuming seeing FWHM=41 pc), and 2. the correlated variance
	structure, as fit from the last model.
	
	Parameters
	----------
	
	Hii_df: pd dataframe
		Contains location and metallicity data about all Hii regions in a galaxy
		Must contain:
			* RA
			* DEC
			* Z_<diag>
			* e_Z_<diag>
			
	diag: str
		Name of the metallicity diagnostic to be used here.
		
	meta: dict
		General facts about the galaxy. Comes from metadata_m83() function usually.
		Must include details about: 
			* D to galaxy
			* i, PA of galaxy
			* Galaxy central coordinates (RA, DEC)
			
	powerlaw_parameters: tuple
		(A, beta, cutoff)
		Comes from last fit of the metallicity model
		
	Returns
	-------
	
	dist_matrix: (N, N) numpy array
		physical distance of each data point from all others (kpc)
	
	Z_resid: (N,) numpy array
		difference between model and observed metallicity, for all data points
	
	fit_params: tuple
		(Z_centre, Z_grad)
	
	fit_uncertainty: tuple
		1 sigma errors in (Z_centre, Z_grad)
	'''
	# Remove nans
	wanted_spaxels = ~np.isnan(Hii_df['Z_'+diag])
	# Find all Hii spaxels' distance from centre
	r_kpc  = RA_DEC_to_radius(Hii_df['RA'][wanted_spaxels], Hii_df['DEC'][wanted_spaxels], meta)
	# Find all Hii spaxels' distance from each other
	dist_matrix = deprojected_distances(Hii_df['RA'][wanted_spaxels], Hii_df['DEC'][wanted_spaxels], meta=meta)
	measurement_cov = build_error_covariance_matrix(dist_matrix, seeing=0.041, e_Z=Hii_df['e_Z_'+diag][wanted_spaxels])
	true_cov		= powerlaw_cutoff_cov_matrix(dist_matrix, *powerlaw_parameters)
	# Compute linear regression using GLS
	covariates = np.array([np.ones(len(r_kpc)), r_kpc]).T 
	Z_grad_model = GLS(Hii_df['Z_'+diag][wanted_spaxels], covariates, sigma=true_cov+measurement_cov).fit()
	Z_centre, Z_grad = Z_grad_model.params
	e_Z_centre = np.sqrt(Z_grad_model.cov_params())['const']['const']
	e_Z_grad   = np.sqrt(Z_grad_model.cov_params())['x1']['x1']
	return dist_matrix, np.array(Z_grad_model.resid), (Z_centre, Z_grad), (e_Z_centre, e_Z_grad)

def compute_Z_grad_LTE(Hii_df, diag, meta, outliers=True):
	'''
	Uses the LTE method/the LTE_fit package of Cappellari+13.
	Taken from: https://ui.adsabs.harvard.edu/abs/2013MNRAS.432.1709C/exportcitation
	
	Parameters
	----------
	
	Hii_df: pd dataframe
		Contains location and metallicity data about all Hii regions in a galaxy
		Must contain:
			* RA
			* DEC
			* Z_<diag>
			* e_Z_<diag>
			
	diag: str
		Name of the metallicity diagnostic to be used here.
		
	meta: dict
		General facts about the galaxy. Comes from metadata_m83() function usually.
		Must include details about: 
			* D to galaxy
			* i, PA of galaxy
			* Galaxy central coordinates (RA, DEC)
	
	outliers: bool (default=True)
		If True, excludes outliers. 
		If False, does not.
	
	Returns
	-------
	p: the resulting fit.
		Has attributes:
		* p.ab, which gives Z_grad and Z_centre
		* p.ab_err (it's what it sounds like)
		* p.mask, which tells you (T/F) whether something's an outlier.
		* p.sig_int: intrinsic scatter of the relation (assumed to be constant 
		  and uncorrelated)
	'''
	wanted_spaxels = ~np.isnan(Hii_df['Z_'+diag])
	# Find all Hii spaxels' distance from centre
	r_kpc  = RA_DEC_to_radius(Hii_df['RA'][wanted_spaxels], Hii_df['DEC'][wanted_spaxels], meta)
	Z = Hii_df['Z_'+diag][wanted_spaxels]
	# Assume uncorrelated errors
	e_Z=Hii_df['e_Z_'+diag][wanted_spaxels]
	if outliers:
		p = lts_linefit(np.array(r_kpc), np.array(Z), np.zeros_like(r_kpc), np.array(e_Z), pivot=np.median(r_kpc),plot=False)
	else:
		p = lts_linefit(np.array(r_kpc), np.array(Z), np.zeros_like(r_kpc), np.array(e_Z), pivot=np.median(r_kpc), clip=10000)
	return p, np.median(r_kpc)

def picture_LTE_outliers(Hii_df, diag, result):
	wanted_spaxels = ~np.isnan(Hii_df['Z_'+diag])
	RA_list	 = Hii_df['RA'][wanted_spaxels]
	DEC_list = Hii_df['DEC'][wanted_spaxels]
	plt.scatter(x=RA_list[result.mask], y=DEC_list[result.mask],s=1,c='blue',label="Fitted points")
	plt.scatter(x=RA_list[~result.mask], y=DEC_list[~result.mask],s=2,c='pink',label="Outliers")
	plt.xlabel('RA')
	plt.ylabel('DEC')
	plt.legend()
	plt.show()
	return 1

def make_dist_matrix_and_Z_resid_GLS(Hii_df, diag, meta):
	# Remove nans
	wanted_spaxels = ~np.isnan(Hii_df['Z_'+diag])
	# Find all Hii spaxels' distance from centre
	r_kpc  = RA_DEC_to_radius(Hii_df['RA'][wanted_spaxels], Hii_df['DEC'][wanted_spaxels], meta)
	# Find all Hii spaxels' distance from each other
	dist_matrix = deprojected_distances(Hii_df['RA'][wanted_spaxels], Hii_df['DEC'][wanted_spaxels], meta=meta)
	e_Z = build_error_covariance_matrix(dist_matrix, seeing=0.041, e_Z=Hii_df['e_Z_'+diag][wanted_spaxels])
	# Compute linear regression using GLS
	covariates = np.array([np.ones(len(r_kpc)), r_kpc]).T 
	Z_grad_model = GLS(Hii_df['Z_'+diag][wanted_spaxels], covariates, sigma=e_Z).fit()
	return dist_matrix, np.array(Z_grad_model.resid)
	
	
# UPDATE RIGHT NOW:
def make_dist_matrix_and_Z_resid_from_df(Hii_df, diag, meta, Z_model):
	# Remove nans
	wanted_spaxels = ~np.isnan(Hii_df['Z_'+diag])
	# Find all Hii spaxels' distance from centre
	r_kpc  = RA_DEC_to_radius(Hii_df['RA'][wanted_spaxels], Hii_df['DEC'][wanted_spaxels], meta)
	# Find all Hii spaxels' distance from each other
	dist_matrix = deprojected_distances(Hii_df['RA'][wanted_spaxels], Hii_df['DEC'][wanted_spaxels], meta=meta)
	# find residual Zs.
	Z_resid = np.array(Hii_df['Z_'+diag][wanted_spaxels]) - Z_model[0] - Z_model[1]*np.array(r_kpc)
	return dist_matrix, Z_resid
	
def get_semivariogram(Z, dist_matrix, bin_size = 0.1, f_keep=1.0):
	'''
	Computes the empirical semivariogram in each bin
	
	Parameters
	----------
	Z : array-like (m,)
		The variable we are making a semivariogram of. 
		Usually residuals in metallicity.
	
	dist_matrix: np array (m, m)
		Gives the distance from all pairs of points.
		Comes from deprojected_distances, so units are kpc.
		
	bin_size: float
		size of bins. Same units as dist_matrix
		Here I'll default to 100 pc
		
	f_keep: float
		Famously semivariograms get unreliable as you go out to high distances
		But for right now I'm into that.
		So keep all of them (default to 1)
		
	Returns
	-------
	semivariogram: array
		Value of empirical semivariogram in each bin
	
	bins: array
		Location of the bins. For plotting.
	
	n_pairs_per_bin: array
		# of pairs of data points used to compute the semivariance in each bin.
		Good for diagnostics - maybe you want to cut off after the first bin
		with n_pairs < 20 or so?
	'''
	m = len(Z)
	bins = np.arange(0, np.max(np.max(dist_matrix)), bin_size)
	dif_pairs = [[] for x in bins]
	for i in range(m):
		for j in range(i+1,m):
			bin_index = int(np.floor(dist_matrix[i,j]/bin_size))
			dif_pairs[bin_index].append(Z[i]-Z[j])
	# Now get variance for each bin
	semivariogram = 0.5 * np.array([np.var(x, ddof=1) for x in dif_pairs])
	n_pairs_per_bin = [len(x) for x in dif_pairs]
	max_i = int(len(bins)*f_keep)
	return semivariogram[:max_i], bins[:max_i], n_pairs_per_bin[:max_i]	   
	
def get_weighted_semivariogram(Z, e_Z, dist_matrix, bin_size = 0.1, f_keep=1.0):
	'''
	Computes the empirical semivariogram in each bin, using the weighted 
	variance estimation formula
	
	Parameters
	----------
	Z : array-like (m,)
		The variable we are making a semivariogram of. 
		Usually residuals in metallicity.
	
	e_Z : array-like (m,)
		Uncertainties (s.d.) in the variable Z
	
	dist_matrix: np array (m, m)
		Gives the distance from all pairs of points.
		Comes from deprojected_distances, so units are kpc.
		
	bin_size: float
		size of bins. Same units as dist_matrix
		Here I'll default
		
	f_keep: float
		Famously semivariograms get unreliable as you go out to high distances
		But for right now I'm into that.
		So keep all of them (default to 1)
		
	Returns
	-------
	semivariogram: array
		Value of empirical semivariogram in each bin
	
	bins: array
		Location of the bins. For plotting.
	
	n_pairs_per_bin: array
		# of pairs of data points used to compute the semivariance in each bin.
		Good for diagnostics - maybe you want to cut off after the first bin
		with n_pairs < 20 or so?
	'''
	m = len(Z)
	bins = np.arange(0, np.max(np.max(dist_matrix)), bin_size)
	dif_pairs = [[] for x in bins]
	weights = [[] for x in bins]
	# for each pair, compute difference between pairs, and variance in this.
	for i in range(m):
		for j in range(i+1,m):
			bin_index = int(np.floor(dist_matrix[i,j]/bin_size))
			dif_pairs[bin_index].append(Z[i]-Z[j])
			weights[bin_index].append(1.0/(e_Z[i]**2 + e_Z[j]**2) )
			# this is 1/var, computed with linear error propagation
			# Any error in the model is 100% correlated, so only measurement errors
			# should account for this weight term. 
	semivariogram = 0.5 * np.array([weighted_var(dif_pairs[i], weights[i]) for i in range(len(dif_pairs))])
	n_pairs_per_bin = [len(x) for x in dif_pairs]
	max_i = int(len(bins)*f_keep)
	return semivariogram[:max_i], bins[:max_i], n_pairs_per_bin[:max_i]	  
	
def krig_model(RA, DEC, Hii_df, meta, model_params, diag, mode='grid',model='exp', nugget=False):
	##### NOTE! Only tested for grid so far.
	'''
	Performs universal kriging on a model grid of RA and DEC
	Uses my distance function, KT18's covariance function, the best fitting
	value of f_d (no restrictions), and assumes Z ~ r + (random effects)
	
	Parameters
	----------
	RA: (M,) np array
		List of RA values. Must be in degrees.
		
	DEC: (N,) np array
		List of DEC values. Must be in degrees.	   
		
	Hii_df: data frame, containing RA, DEC, r, Z, e_Z for each data point.
	
	meta: dict
		Metadata used to calculate the distances. Must contain:
		PA: float
			Principle Angle of the galaxy, degrees.
		i: float
			inclination of the galaxy along this principle axis, degrees.
		D: float
			Distance from this galaxy to Earth, Mpc.
	
	mode: str
		Options include
		'grid': make a grid of all possible combos of supplied RA and DEC
				values; use kriging to estimate Z at each point on grid.
		'list': just use pairs of RA and DEC values as they are given.
				RA and DEC must have the same length.
		'auto': Get RA and DEC values from the df itself.
		
	diag: str
		Which metallicity diagnostic are you going to use?
		Accepted diags are: O3N2, N202, N2Ha, R23, N2S2.
	
	Returns
	-------
	Z_pred_matrix: (M,N) np array
	grid of interpolated (kriged) values over the RA, DEC coords given.
	
	var_matrix: (M,N) np array
	variances for these predictions
	'''
	check_diag(diag)
	if mode=='grid':
		# Construct arrays for all pairs of RA and DEC values
		RA_grid, DEC_grid = np.meshgrid(RA, DEC)
		RA_long = np.concatenate(RA_grid)
		DEC_long = np.concatenate(DEC_grid)
	elif mode=='list':
		RA_long = RA
		DEC_long = DEC
	elif mode=='auto':
		RA_long = df['RAdeg']
		DEC_long = df['DEdeg']
	else:
		print("Error: Bad argument given to `krig_model`.\nMode must be either 'grid', 'list', or 'auto'.")
		return np.nan, np.nan
	############################################################################
	#																		   #
	#						 Construct covariance matrices					   #
	#																		   #
	############################################################################
	data_dists, Z_resids = make_dist_matrix_and_Z_resid_from_df(Hii_df, diag, meta, model_params)
	Hii_RAs, Hii_DECs, _ , e_Z = unpack_and_trim(Hii_df, diag)
	data_grid_dists = deprojected_distances(Hii_RAs, Hii_DECs, RA_long, DEC_long, meta=meta)
	# Find galactocentric radius of each grid point and each data point
	grid_r = deprojected_distances(RA_long, DEC_long, meta['RA'], meta['DEC'], meta=meta).T[0]
	Hii_r  = RA_DEC_to_radius(Hii_RAs, Hii_DECs, meta)
	covariates = np.array([np.ones(len(Hii_r)), Hii_r]).T 
	grid_covariates = np.array([np.ones(len(grid_r)), grid_r]).T
	best_beta = np.array([model_params['Z_c'][diag], model_params['Z_grad (kpc^-1)'][diag]])
	error_cov  = build_error_covariance_matrix(data_dists, e_Z, meta)
	if nugget:
		nugget_cov = np.diag( model_params['nugget'][diag]*np.ones(len(e_Z)) ) 
	else:
		nugget_cov=0
	if model=='powerlaw_cutoff':
		A = 10.0**model_params['log_Var'][diag]
		power_index = model_params['beta'][diag]
		cutoff = model_params['phi'][diag]
		spatial_cov_data = powerlaw_cutoff_cov_matrix(data_dists, A, power_index, cutoff)
		spatial_cov_data_grid = powerlaw_cutoff_cov_matrix(data_grid_dists, A, power_index, cutoff)
	elif model=='exp' or model=='exponential':
		A = 10.0**model_params['log_Var'][diag]
		phi = model_params['phi'][diag]
		spatial_cov_data = A*exp_corr_matrix(data_dists, phi)
		spatial_cov_data_grid = A*exp_corr_matrix(data_grid_dists, phi)
	tot_data_cov = spatial_cov_data + error_cov + nugget_cov
	# Use Universal Kriging to estimate \eta for each grid point
	# SpaceTimeWithR, equation 4.6	
	c_factor = cho_factor(tot_data_cov)
	white_resids = cho_solve(c_factor, Z_resids)
	predicted_grid_resids = spatial_cov_data_grid.T @ white_resids
	# Add this to model to predict metallicity at each point.
	predicted_grid_Z = np.dot(best_beta, grid_covariates.T) + predicted_grid_resids
	# Get uncertainty (eq. 4.10 of SpaceTimeWithR)
	white_D = cho_solve(c_factor, covariates)
	white_cov_data_grid = cho_solve(c_factor, spatial_cov_data_grid)
	Cov_cvars = covariates.T @ white_D
	inv_cov_cvars =np.linalg.inv(Cov_cvars) # don't be fancy, since it's a 2 by 2 
	kriged_cvar_resids = grid_covariates - (spatial_cov_data_grid.T @ white_D)
	gls_uncertainty = np.diagonal( kriged_cvar_resids @ inv_cov_cvars @ kriged_cvar_resids.T)
	grid_uncertainty = A - np.diagonal(spatial_cov_data_grid.T @ white_cov_data_grid) + gls_uncertainty
	
	# If needed, reshape these to be an RA X DEC shaped, plottable matrix
	if mode=='grid':
		Z_pred_matrix = predicted_grid_Z.reshape(len(RA), len(DEC))
		var_matrix = grid_uncertainty.reshape(len(RA), len(DEC))
		return Z_pred_matrix, var_matrix
	else:
		return predicted_grid_Z, grid_uncertainty
		
def d_from_closest_points(RA, DEC, Hii_df, meta, diag):
	'''
	From a list of RA/DEC points, find the distance from each point to its
	closest Hii region. Report this in kpc.
	'''
	RA_grid, DEC_grid = np.meshgrid(RA, DEC)
	RA_long = np.concatenate(RA_grid)
	DEC_long = np.concatenate(DEC_grid)
	Hii_RAs, Hii_DECs, _ , e_Z = unpack_and_trim(Hii_df, diag)
	data_grid_dists = deprojected_distances(Hii_RAs, Hii_DECs, RA_long, DEC_long, meta=meta)
	d_to_closest_HII_regions = np.min(data_grid_dists, axis = 0)
	min_dist_matrix = d_to_closest_HII_regions.reshape(len(RA), len(DEC))
	return min_dist_matrix
	
		
def krig_predict(RA, DEC, Hii_RAs, Hii_DECs, Z, e_Z, meta, theta, beta, mode='list', model='exp'):
	'''
	Slightly restructured version of above, to be compatible with ten_fold_CV program.
	
	Parameters
	----------
	RA: (M,) np array
		List of RA values. Must be in degrees.
		
	DEC: (N,) np array
		List of DEC values. Must be in degrees.	  
	
	Hii_RA, Hii_DEC, Z, e_Z: np arrays
		Unpacked versions of Hii_df.
		Allows for more flexibility	 -- i.e. we can select only a subsample of df points.
	
	meta: dict
		Metadata used to calculate the distances. Must contain:
		PA: float
			Principle Angle of the galaxy, degrees.
		i: float
			inclination of the galaxy along this principle axis, degrees.
		D: float
			Distance from this galaxy to Earth, Mpc.
			
	theta: tuple of model params, for small-scale covariance structure.
	
	beta: tuple of model params, for large scale trends
		  beta[0] is Z_char -- characteristic metallicity at r_char
		  beta[1] is r_char -- metallicity at which Z_char is computed.
		  beta[2] is Z_grad (units: dex kpc^-1)
	
	mode: str
		Options include
		'grid': make a grid of all possible combos of supplied RA and DEC
				values; use kriging to estimate Z at each point on grid.
		'list': just use pairs of RA and DEC values as they are given.
				RA and DEC must have the same length.
		'auto': Get RA and DEC values from the df itself.
	
	model: str
		What is the small-scale covariance structure?
		Currently, only exp is implemented.
	
	Returns
	-------
	Z_pred_matrix: np array
	interpolated (kriged) values over the RA, DEC coords given.
	
	var_matrix: np array
	variances for these predictions
	'''
	if mode=='grid':
		# Construct arrays for all pairs of RA and DEC values
		RA_grid, DEC_grid = np.meshgrid(RA, DEC)
		RA_long = np.concatenate(RA_grid)
		DEC_long = np.concatenate(DEC_grid)
	elif mode=='list':
		RA_long = RA
		DEC_long = DEC
	elif mode=='auto':
		RA_long = df['RAdeg']
		DEC_long = df['DEdeg']
	else:
		print("Error: Bad argument given to `krig_model`.\nMode must be either 'grid', 'list', or 'auto'.")
		return np.nan, np.nan
	if model != 'exp':
		print("Apologies; only the model 'exp' is currently implemented.")
		print("If you are not Benji Metha, email methab@student.edu.au and he will patch this.")
		print("If you are Benji Metha, then it's time to fix this.")
		exit(1)
		return None
	############################################################################
	#																		   #
	#						 Construct covariance matrices					   #
	#																		   #
	############################################################################
	Z_char, r_char, Z_grad = beta
	Hii_r  = RA_DEC_to_radius(Hii_RAs, Hii_DECs, meta)
	covariates = np.array([np.ones(len(Hii_r)), Hii_r - r_char]).T 
	Z_resids = Z - Z_char - Z_grad*(Hii_r - r_char)
	data_dists = deprojected_distances(Hii_RAs, Hii_DECs, meta=meta)
	data_grid_dists = deprojected_distances(Hii_RAs, Hii_DECs, RA_long, DEC_long, meta=meta)
	grid_r = deprojected_distances(RA_long, DEC_long, meta['RA'], meta['DEC'], meta=meta).T[0]
	grid_covariates = np.array([np.ones(len(grid_r)), grid_r - r_char]).T
	error_cov  = build_error_covariance_matrix(data_dists, e_Z, meta)
	best_beta = np.array([Z_char, Z_grad])
	# Spatial Covariances
	A = 10.0**theta[0]
	phi = theta[1]
	spatial_cov_data = A*exp_corr_matrix(data_dists, phi)
	spatial_cov_data_grid = A*exp_corr_matrix(data_grid_dists, phi)
	tot_data_cov = spatial_cov_data + error_cov
	############################################################################
	#																		   #
	#							  Universal Kriging							   #
	#																		   #
	############################################################################
	# Use Universal Kriging to estimate \eta for each grid point
	# SpaceTimeWithR, equation 4.6	
	c_factor = cho_factor(tot_data_cov)
	white_resids = cho_solve(c_factor, Z_resids)
	predicted_grid_resids = spatial_cov_data_grid.T @ white_resids
	# Add this to model to predict metallicity at each point.
	predicted_grid_Z = np.dot(best_beta, grid_covariates.T) + predicted_grid_resids
	# Get uncertainty (eq. 4.10 of SpaceTimeWithR)
	white_D = cho_solve(c_factor, covariates)
	white_cov_data_grid = cho_solve(c_factor, spatial_cov_data_grid)
	Cov_cvars = covariates.T @ white_D
	inv_cov_cvars =np.linalg.inv(Cov_cvars) # don't be fancy, since it's a 2 by 2 
	kriged_cvar_resids = grid_covariates - (spatial_cov_data_grid.T @ white_D)
	gls_uncertainty = np.diagonal( kriged_cvar_resids @ inv_cov_cvars @ kriged_cvar_resids.T)
	grid_uncertainty = A - np.diagonal(spatial_cov_data_grid.T @ white_cov_data_grid) + gls_uncertainty
	# If needed, reshape these to be an RA X DEC shaped, plottable matrix
	if mode=='grid':
		Z_pred_matrix = predicted_grid_Z.reshape(len(RA), len(DEC))
		var_matrix = grid_uncertainty.reshape(len(RA), len(DEC))
		return Z_pred_matrix, var_matrix
	else:
		return predicted_grid_Z, grid_uncertainty
	