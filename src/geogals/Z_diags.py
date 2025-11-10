'''
Z_diags.py

Trimmed to include only 3 metallicity diagnostics -- N2S2Ha, Scal, and O3N2

Shaped to run over the PHANGS emission line map files (HDU lists)

Created by: Benjamin Metha
Last Updated: Feb 19, 2025
'''

import numpy as np 
Z_sun = 8.69
data_path = '../Data/Handmade/Diag_data/'

# Import memoisations
if __name__ != '__main__':
	O3N2_cal_data = np.loadtxt(data_path + 'Curti17_O3N2.txt')

def compute_Z(line_df, diag):
	'''
	Master function to help parallelise between different diagnostics.
	
	Takes in a line_df and a diagnostic.
	
	Computes metallicity map using that line diagnostic.
	
	Parameters
	----------
	line_df: hdu list
		A big guy containing all the different emission line data reduced
		from TYPHOON data cubes
	
	diag: str
		The name that is given to the metallicity diagnostic used for this
		computation. May be upper or lower case.
	
	Returns
	-------
	Z: array
		Metallicity using this diagnostic
		
	e_Z: array
		Error in metallicity using this diagnostic, 
		computed via linear error propagation.
	'''
	diag = diag.upper()
	if diag == 'N2S2HA':
		return compute_Z_N2S2Ha_Dop16(line_df)
	elif diag == 'O3N2':
		return compute_Z_O3N2_Curti17(line_df)
	elif diag == 'SCAL':
		return compute_Z_Scal_Dop16(line_df)
	else:
		print("Error: " + diag + " seems not to be a metallicity diagnostic I have implemented!")
		return None, None

def compute_Z_N2S2Ha_Dop16(line_df):
	'''
	Given a set of deredenned emission line maps+error, 
	compute metallicity maps+error, using the
	N2S2Ha diagnostic of Dopita+2016:
	https://ui.adsabs.harvard.edu/abs/2016Ap&SS.361...61D
	
	Parameters
	----------
	line_df: hdu list
		A big guy containing all the different emission line data reduced
		from TYPHOON data cubes
	
	Returns
	-------
	Z: array
		Metallicity using this diagnostic
		
	e_Z: array
		Error in metallicity using this diagnostic, 
		computed via linear error propagation.
	'''
	# Unpack the wanted lines: f_NII, f_SII6717, f_SII6731, f_Ha,
	# and their errors.
	line_IDs = [line_df[x].header['EXTNAME'] for x in range(len(line_df))]
	# Here, NII is Nii 6584
	# [NII]λ6584/[NII]λ6548 is 2.92:1 (Acker 1989)
	f_NII = line_df[line_IDs.index('NII6583_FLUX')].data
	e_NII = line_df[line_IDs.index('NII6583_FLUX_ERR')].data
	f_SII6716 = line_df[line_IDs.index('SII6716_FLUX')].data
	e_SII6716 = line_df[line_IDs.index('SII6716_FLUX_ERR')].data
	f_SII6731 = line_df[line_IDs.index('SII6730_FLUX')].data
	e_SII6731 = line_df[line_IDs.index('SII6730_FLUX_ERR')].data
	f_Ha = line_df[line_IDs.index('HA6562_FLUX')].data
	e_Ha = line_df[line_IDs.index('HA6562_FLUX_ERR')].data
	# compute relevant line ratios
	N2S2  = np.log10(f_NII/(f_SII6716+f_SII6731) )
	N2Ha  = np.log10(f_NII/f_Ha)
	N2S2Ha = N2S2 + 0.264*N2Ha
	Z_N2S2Ha_low = 8.77 + N2S2Ha
	Z_N2S2Ha_upper_correction = 0.45 * (N2S2Ha+0.3)**5
	Z_N2S2Ha = Z_N2S2Ha_low + Z_N2S2Ha_upper_correction*(Z_N2S2Ha_low > 9.05)
	# and errors
	dZ_dN2S2Ha = 1 + 2.25*(N2S2Ha+0.3)**4 *(Z_N2S2Ha_low > 9.05)
	dratio_dN2 = 1.264/(np.log(10)*f_NII)
	dratio_dS2 = 1.0/(np.log(10)*(f_SII6716+f_SII6731))
	dratio_dHa = 0.264/(np.log(10)*f_Ha)
	e_Z2 = dZ_dN2S2Ha**2 * ((dratio_dN2*e_NII)**2 + (dratio_dS2*e_SII6716)**2 \
							+ (dratio_dS2*e_SII6731)**2 + (dratio_dHa*e_Ha)**2)
	e_Z	 = np.sqrt(e_Z2)
	return Z_N2S2Ha, e_Z
	 
def compute_Z_O3N2_Curti17(line_df):
	'''
	Given a set of deredenned emission line maps+error, 
	compute metallicity maps+error, using the
	N2S2Ha diagnostic of Curti+2017:
	https://ui.adsabs.harvard.edu/abs/2017MNRAS.465.1384C/abstract
	
	Optionally, add a DIG correction as devised by Kumari+19:
	https://ui.adsabs.harvard.edu/abs/2019MNRAS.485..367K
	
	Parameters
	----------
	line_df: hdu list
		A big guy containing all the different emission line data reduced
		from TYPHOON data cubes

	Returns
	-------
	Z: array
		Metallicity using this diagnostic
		
	e_Z: array
		Error in metallicity using this diagnostic, 
		computed via linear error propagation.
	'''
	# Unpack the wanted lines: f_NII, f_OIII, f_Ha, f_Hb,
	# and their errors.
	line_IDs = [line_df[x].header['EXTNAME'] for x in range(len(line_df))]
	f_NII  = line_df[line_IDs.index('NII6583_FLUX')].data
	e_NII  = line_df[line_IDs.index('NII6583_FLUX_ERR')].data
	f_OIII = line_df[line_IDs.index('OIII5006_FLUX')].data
	e_OIII = line_df[line_IDs.index('OIII5006_FLUX_ERR')].data
	f_Ha   = line_df[line_IDs.index('HA6562_FLUX')].data
	e_Ha   = line_df[line_IDs.index('HA6562_FLUX_ERR')].data
	f_Hb   = line_df[line_IDs.index('HB4861_FLUX')].data
	e_Hb   = line_df[line_IDs.index('HB4861_FLUX_ERR')].data
	# define line ratios
	O3	   = np.log10(f_OIII/f_Hb) 
	N2	   = np.log10(f_NII/f_Ha)
	O3N2   = O3 - N2 
	# compute Z_O3N2 by interpolating the inverse function
	Z_o3n2 = np.interp(O3N2,O3N2_cal_data[:,0],O3N2_cal_data[:,1],left=np.nan, right=np.inf)
	x_o3n2 = Z_o3n2 - Z_sun
	# compute error function
	dZ_dratio= 1.0/(4.765+4.536*x_o3n2)
	e_Z2 = (dZ_dratio/np.log(10))**2 * ((e_NII/f_NII)**2 + (e_OIII/f_OIII)**2 + (e_Hb/f_Hb)**2+ (e_Ha/f_Ha)**2)
	e_Z = np.sqrt(e_Z2)
	return Z_o3n2, e_Z

# TODO rename, and push change forward!

def compute_Z_Scal_Dop16(line_df):
	'''
	Given a set of deredenned emission line maps+error, 
	compute metallicity maps+error, using the
	Scal diagnostic of Pilyugin and Grebel 2016:
	https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.3678P/abstract
	
	Parameters
	----------
	line_df: hdu list
		A big guy containing all the different emission line data reduced
		from TYPHOON data cubes
	
	Returns
	-------
	Z: array
		Metallicity using this diagnostic
		
	e_Z: array
		Error in metallicity using this diagnostic, 
		computed via linear error propagation.
	'''
	line_IDs = [line_df[x].header['EXTNAME'] for x in range(len(line_df))]
	# Get all lines and errors
	f_Hb   = line_df[line_IDs.index('HB4861_FLUX')].data
	e_Hb   = line_df[line_IDs.index('HB4861_FLUX_ERR')].data
	f_SII6716 = line_df[line_IDs.index('SII6716_FLUX')].data
	e_SII6716 = line_df[line_IDs.index('SII6716_FLUX_ERR')].data
	f_SII6731 = line_df[line_IDs.index('SII6730_FLUX')].data
	e_SII6731 = line_df[line_IDs.index('SII6730_FLUX_ERR')].data
	# [NII]λ6584/[NII]λ6548 is 2.92:1 (Acker 1989)
	f_NII = (1+1/2.92)*line_df[line_IDs.index('NII6583_FLUX')].data # NEED TO ADD 6548?
	e_NII = (1+1/2.92)*line_df[line_IDs.index('NII6583_FLUX_ERR')].data
	# [OIII]λ5007/[OIII]λ4959 is 2.98:1 (Storey & Zeippen 2000)
	f_OIII = (1+1/2.98)*line_df[line_IDs.index('OIII5006_FLUX')].data # NEED TO ADD 4959?
	e_OIII = (1+1/2.98)*line_df[line_IDs.index('OIII5006_FLUX_ERR')].data
	
	# Define N2, S2, and R3.
	N2 = f_NII / f_Hb
	S2 = (f_SII6716 + f_SII6731)/f_Hb
	R3 = f_OIII/f_Hb
	
	# Using linear error propagation, get uncertainties on these from s.d. of
	# all lines
	e_S2 = np.sqrt( e_SII6716**2 + e_SII6731**2 + (S2*e_Hb)**2 ) / f_Hb
	e_N2 = np.sqrt( e_NII**2  + (N2*e_Hb)**2 ) / f_Hb # NEED TO ADD 6548?
	e_R3 = np.sqrt( e_OIII**2 + (R3*e_Hb)**2 ) / f_Hb # NEED TO ADD 4959?
	
	Z_Scal_upper =	8.424  + 0.030*np.log10(R3/S2) + 0.762*np.log10(N2) \
				 + (-0.349 + 0.182*np.log10(R3/S2) + 0.508*np.log10(N2) )*np.log10(S2)
	
	Z_Scal_lower =	8.072 + 0.789*np.log10(R3/S2)  + 0.762*np.log10(N2) \
				 + (1.069 - 0.170*np.log10(R3/S2)  + 0.022*np.log10(N2) )*np.log10(S2)
	
	# Compute partial derivatives for linear error propagation
			 
	dZup_dN2  = (0.762  + 0.508*np.log10(S2))   * np.log10(np.e) / N2
	dZlow_dN2 = (0.762  + 0.022*np.log10(S2))   * np.log10(np.e) / N2
	dZup_dS2  = (-0.030 - 2*0.182*np.log10(S2)) * np.log10(np.e) / S2
	dZlow_dS2 = (-0.789 + 2*0.170*np.log10(S2)) * np.log10(np.e) / S2
	dZup_dR3  = (0.030  + 0.182*np.log10(S2))   * np.log10(np.e) / R3
	dZlow_dR3 = (0.789  - 0.170*np.log10(S2))   * np.log10(np.e) / R3
	
	# Compute uncertainty on upper and lower branch
	
	e_Z_Scal_upper = np.sqrt( (dZup_dN2*e_N2)**2  + (dZup_dS2*e_S2)**2	+ (dZup_dR3*e_R3)**2 )
	e_Z_Scal_lower = np.sqrt( (dZlow_dN2*e_N2)**2 + (dZlow_dS2*e_S2)**2 + (dZlow_dR3*e_R3)**2 )
	
	in_upper	 = (np.log10(N2) >= -0.6)
	in_lower	 = 1 - in_upper
	
	# Combine the two branches for an Scal metallicity
	Z_Scal = Z_Scal_lower*in_lower + Z_Scal_upper*in_upper
	# Note: We are not accounting for any error in whether or not the metallicity
	# should be calculated for the upper or lower branch
	e_Z_Scal = e_Z_Scal_lower*in_lower + e_Z_Scal_upper*in_upper
	
	return Z_Scal, e_Z_Scal

if __name__=='__main__':
	# Run this to generate/save memoisations.
	x = np.arange(7.6, 8.9, 0.01) - Z_sun 
	# for accuracies of 0.01 in Zworld
	# Range over which metallicities are valid taken from Curti+2020.
	O3N2 = 0.281 -4.765*x -2.268*(x**2)
	O3N2_cal_data = np.vstack((O3N2[::-1], x[::-1]+Z_sun)).T
	np.savetxt(data_path+'Curti17_O3N2.txt',O3N2_cal_data)
	# Aaaaand for O3S2
	O3S2_2017 = -0.046 -2.223*x -1.073*(x**2) + 0.533* (x**3) 
	O3S2_cal_data_2017 = np.vstack((O3S2_2017[::-1], x[::-1]+Z_sun)).T
	np.savetxt(data_path+'Curti17_O3S2.txt',O3S2_cal_data_2017)
	O3S2_2020 = -0.054 -2.546*x -1.970*(x**2) + 0.082* (x**3) + 0.222*(x**4)
	O3S2_cal_data_2020 = np.vstack((O3S2_2020[::-1], x[::-1]+Z_sun)).T
	np.savetxt(data_path+'Curti20_O3S2.txt',O3S2_cal_data_2020)