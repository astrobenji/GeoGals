'''
A submodule featuring functions designed to generate metallicity maps from
PHANGS data.

Created by: Benjamin Metha

Last updated: May 15, 2025
'''

######################
#    Preprocessing   #
######################

# Caution: these functions are PHANGS specific and may not work in general

def SN_cut(line_df, threshold=3):
	'''
	Replace all spaxels with SN<3 in a certain line with NANs.

	Parameters
	----------
	lines_df: hdu list
		A big guy containing all the different emission line data
		present in PHANGS maps files

	threshold: float
		At what S/N do we cut a line? (Defaulted to 3)

	Returns
	-------
	lines_df: hdu list
		The same hdu list, but with lines where S/N < threshold
		replaced with np.nan
	'''
	n_lines = 8 # for PHANGS data
	x_max, y_max = line_df[30].data.shape
	for l in range(1, n_lines+1):
		signal = line_df[6*l - 1].data
		noise  = line_df[6*l].data
		too_low = signal <= threshold*noise
		# replace low signals/no signals with NANs.
		for ii in range(x_max):
			for jj in range(y_max):
				if too_low[ii,jj]:
					signal[ii,jj] = np.nan
					noise[ii,jj]  = np.nan
	return line_df

def extinction_correction(line_df, wavelengths, R_V=3.1):
	'''
	Parameters
	----------

	lines_df: hdu list
		A big guy containing all the different emission line data
		present in PHANGS maps files

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
	Ha_map = line_df[line_IDs.index('HA6562_FLUX')].data
	Hb_map = line_df[line_IDs.index('HB4861_FLUX')].data
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
		line_df[6+l-1].data	 = line_df[6+l-1].data * 10**(0.4 * extinction_map)
		line_df[6*l].data	 = line_df[6*l].data   * 10**(0.4 * extinction_map)

	return line_df

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
	O3Hb = np.log10( line_df[line_IDs.index('OIII5006_FLUX')].data /	 line_df[line_IDs.index('HB4861_FLUX')].data )
	S2Ha = np.log10( (line_df[line_IDs.index('SII6716_FLUX')].data+line_df[line_IDs.index('SII6730_FLUX')].data)/line_df[line_IDs.index('HA6562_FLUX')].data	 )
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
	O3Hb = np.log10( line_df[line_IDs.index('OIII5006_FLUX')].data/line_df[line_IDs.index('HB4861_FLUX')].data )
	N2Ha = np.log10( line_df[line_IDs.index('NII6583_FLUX')].data/line_df[line_IDs.index('HA6562_FLUX')].data	   )
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
