'''
A collection of functions built for the geostatistical analysis of galaxy data.

Created by: Benjamin Metha, Tree Smith, Jaime Blackwell
Last Updated: May 09, 2025
'''

from . import __version__

import numpy as np
#from   astropy.io import fits
from   sklearn.metrics.pairwise import euclidean_distances 
#import pandas as pd 
from   astropy.wcs import WCS
#import astropy.units as u
#from   scipy.linalg import cho_factor, cho_solve
#from   extinction import ccm89, apply
import scipy
import emcee
from statsmodels.regression.linear_model import GLS

ASEC_PER_RAD = 206265.0

#########################
#    Unit Conversions   #
#########################

def make_RA_DEC_grid(header):
	'''
	Given a hdu header, create a grid of RA//DEC for each pixel in that file.
	'''
	world = WCS(header)
	x = np.arange(header['NAXIS1'])
	y = np.arange(header['NAXIS2'])
	X, Y = np.meshgrid(x, y)
	RA_grid, DEC_grid = world.wcs_pix2world(X, Y, 0)
	return RA_grid, DEC_grid
	
def make_physical_lag_grid(header, meta):
	'''
	Given a hdu header, create a grid of RA//DEC for each pixel in that file.
	
	Parameters
	----------
	
	header: hdu header file
		Must contain wcs 
	
	meta: dict
		Must contain RA, DEC of the galaxy centre, and PA, i, and D to get
		the galaxy's absolute units (D should be in units of megaparsecs; PA and
		i should be in units of degrees).
	'''
	# First, convert pixel indices to RA and DEC
	world = WCS(header)
	x = np.arange(-header['NAXIS1'], header['NAXIS1']+1)
	y = np.arange(-header['NAXIS2'], header['NAXIS2']+1)
	X, Y = np.meshgrid(x, y)
	RA_grid, DEC_grid = world.wcs_pix2world(X, Y, 0)
	# Next, convert RA and DEC to physical pc using the meta dict
	delta_RA_deg  = RA_grid  - meta['RA']
	delta_DEC_deg = DEC_grid - meta['DEC']
	PA = np.radians(meta['PA'])
	i  = np.radians(meta['i'])
	# 1: Rotate RA, DEC by PA to get y (major axis direction) and x (minor axis direction)
	x_deg = delta_RA_deg*np.cos(PA)  - delta_DEC_deg*np.sin(PA)
	y_deg = delta_DEC_deg*np.cos(PA) + delta_RA_deg*np.sin(PA)
	# 2: Stretch x values to remove inclination effects
	x_deg = x_deg / np.cos(i)
	# 3: Convert units to kpc
	x_rad = np.radians(x_deg)
	y_rad = np.radians(y_deg)
	x_kpc = x_rad * meta['D'] * 1000
	y_kpc = y_rad * meta['D'] * 1000
	return x_kpc, y_kpc

def RA_DEC_to_XY(RA, DEC, meta):
	'''
	Takes in list of RA, DEC coordinates and transforms them into a 
	list of deprojected XY values, where X and Y are the distances from the 
	galaxy's centre in units of kpc
	
	Parameters
	----------
	
	RA: ndarray like of shape (N,)
		List of RA values
		
	DEC: ndarray like of shape (N,)
		List of DEC values
		
	meta: dict
		Must contain RA, DEC of the galaxy centre, and PA, i, and D to get
		the galaxy's absolute units 
		
	Returns
	-------
	
	XY_kpc: (N,2) ndarray
		Contains X and Y coords of all data points with units of kpc
	'''
	delta_RA_deg  = RA  - meta['RA']
	delta_DEC_deg = DEC - meta['DEC']
	PA = np.radians(meta['PA'])
	i  = np.radians(meta['i'])
	# 1: Rotate RA, DEC by PA to get y (major axis direction) and x (minor axis direction)
	x_deg = delta_RA_deg*np.cos(PA)  - delta_DEC_deg*np.sin(PA)
	y_deg = delta_DEC_deg*np.cos(PA) + delta_RA_deg*np.sin(PA)
	# 2: Stretch x values to remove inclination effects
	x_deg = x_deg / np.cos(i)
	# 3: Convert units to kpc
	x_rad = np.radians(x_deg)
	y_rad = np.radians(y_deg)
	x_kpc = x_rad * meta['Dist'] * 1000
	y_kpc = y_rad * meta['Dist'] * 1000
	XY_kpc = np.stack((x_kpc, y_kpc)).T
	return XY_kpc

def RA_DEC_to_radius(RA, DEC, meta):
	return deprojected_distances(RA, DEC, meta['RA'], meta['DEC'], meta).T[0]

#########################
#   Spatial Statistics  #
#########################
	
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
	
def fast_semivarogram(Z_grid, header=None, meta=None, bin_size=2, f_to_keep=1.0):
    '''
    A fast algorithm for computing the semivariogram of galaxy data.

    Parameters
    ----------
	Z_grid (2d np.array)
		Random field for which we are computing the semivariogram
		
	header: hdu header file
		Must contain wcs.
		If not supplied, semivariogram will be computed in units of pixels, with
		no deprojection.
	
	meta: dict
		Metadata used to calculate the distances. Must contain:
		PA: float
			Principle Angle of the galaxy, degrees.
		i: float
			inclination of the galaxy along this principle axis, degrees.
		D: float
			Distance from this galaxy to Earth, Mpc.
		Must be supplied if header is supplied.
		
	bin_size:
		Size of bins for semivariogram.
		Defaults to 2 (pixels) -- should be changed if using physical separations

	f_to_keep: float
		Fraction of data to keep (defaults to all of it -- but we warn the reader
		that the semivariogram becomes unreliable at large separations)
    
    Returns
    -------
    svg: numpy array
    	Semivariogram of the data at each separation
    
    bc: numpy array
    	centres of each semivariogram bin
    	        
    plt.plot(bc, svg) will generate a plot of the semivariogram.
    '''
    nx, ny = Z_grid.shape # shape
    pad_shape =(2*nx -1, 2*ny-1) #required padding

    M = (~np.isnan(Z_grid)).astype(float) # 1 if non-nan
    F_Z2 = scipy.fft.fft2(Z_grid**2, pad_shape) # FFT of Z^2
    F_M = scipy.fft.fft2(M, pad_shape) # FFT of M
    Z_rev = (Z_grid*M)[::-1,::-1] # reversed Z
    F_Z_rev = scipy.fft.fft2(Z_rev, pad_shape) # FFT of Z (tilde)
    F_Z = scipy.fft.fft2(Z_grid*M, pad_shape) # FFT of Z 
    F_Z2_rev = scipy.fft.fft2(Z_rev**2, pad_shape) # FFT of Z^2 (tilde)
    N_fft = scipy.fft.ifft2(F_M * scipy.fft.fft2(M[::-1, ::-1], pad_shape)).real # N computed using FFT

    gamma = np.fft.ifft2(F_Z2 * (F_M)).real + scipy.fft.ifft2(F_M * F_Z2_rev).real - 2*scipy.fft.fftshift(scipy.fft.ifft2(F_Z * np.conj(F_Z)).real) # unnormalised svg
    gamma = gamma/(2* N_fft) # normalised

    # lags
    if header is not None:
    	if meta is None:
    		raise ValueError("Error: if header is supplied, metadata must also be supplied.")
    	# If header and meta are both given,
    	# Compute physical lag separations for all grid points
    	lag_X, lag_Y = make_physical_lag_grid(header, meta)
    else:
    	# If no header is supplied, compute lags in units of pixels
		lag_x = np.arange(pad_shape[1]) - (Z_grid.shape[1] - 1)  # horizontal shift (cols)
		lag_y = np.arange(pad_shape[0]) - (Z_grid.shape[0] - 1)  # vertical shift (rows)
		lag_X, lag_Y = np.meshgrid(lag_x, lag_y)
    # want the norm of the separation:
    r = (lag_X**2 + lag_Y**2)**0.5
    
    d_lim = np.max(r) * f_to_keep

    # bin by r:
    svg = scipy.stats.binned_statistic(r.flatten(), gamma.flatten(), statistic =np.nanmean, bins=int(d_lim/bin_size), range=(0, d_lim))

    # find bin centres:
    bc = svg.bin_edges
    bc = (bc[1:] + bc[:-1])/2

    return svg.statistic, bc

#########################
#     Model Fitting     #
#########################

def fit_radial_linear_trend(Z_grid, e_Z_grid, header, meta, return_covariances=False):
    '''
    Fits a radial trend to the galaxy data.
    Designed for computing metallicity gradients -- other mean models may be
    required for other galaxy data (e.g. velocities)
    
    Parameters
    ----------
	Z_grid: 2d np.array
		Random field for which we are computing the mean radial trend
		
	e_Z_grid: 2d np.array
		Measured error in value of random field at each separation. Can be None.
		
	header: hdu header file
		Must contain wcs
		
	meta: dict
		Metadata used to calculate the distances. Must contain:
		PA: float
			Principle Angle of the galaxy, degrees.
		i: float
			inclination of the galaxy along this principle axis, degrees.
		D: float
			Distance from this galaxy to Earth, Mpc.
			
	return_covariances: bool
		If True, covariances of parameters will be returned as well.
			
	Returns
	-------
	params: list
		Central value and radial gradient of random field.
    '''	
    RA_grid, DEC_grid = make_RA_DEC_grid(header)
    r = RA_DEC_to_radius(RA_grid, DEC_grid, meta)
    covariates = np.array([np.ones(len(r)), r]).T 
    Z_grad_model = GLS(Z, covariates, sigma=e_Z).fit()
    if return_covariances:
    	return Z_grad_model.params, Z_grad_model.normalized_cov_params
    else:
	    return Z_grad_model.params



# Fit a model for the small-scale structure of a galaxy

# Validate a model using N-fold cross-validation

# Krig using data and a model to predict metallicity at an unknown location
