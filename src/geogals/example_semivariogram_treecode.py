'''
Made by Tree Smith, July 11 2025
'''

import geogals as gg
import numpy as np
import scipy

from astropy.io  import fits
from astropy.wcs import WCS

def semivariogram_fft(Z_grid, box_size, bins, d_lim):

    '''
    
    fft semivariogram- updated 20/5/25 to fix normalisation problem
                        updated 28/5/25 to fix lag problem
    Parameters
        Z_grid (2d np.array)
            regular grid of Z values

        box_size list of length 2:
            physical size of grid in x and y directions
        
        bins:
            number of bins for svg

        d_lim:
            upper cutoff value for svg bins
    
    Returns
        svg and distance bins (units are whatever your box_size argument)

            

    '''

    # set up steps (mask and padding)

    nx, ny = Z_grid.shape # shape
    pad_shape =(2*nx -1, 2*ny-1) #required padding
    M_mask = (np.isnan(Z_grid)) # mask
    Z_copy = np.zeros_like(Z_grid)


    # Z_grid[M_mask] = 0 # set nans to 0
    M = (~M_mask).astype(float) # 1 if non-nan
    Z_copy[~M_mask] = Z_grid[~M_mask]

    lag_x = np.arange(pad_shape[0]) - (Z_grid.shape[0] - 1)  # vertical shift (rows)
    lag_y = np.arange(pad_shape[1]) - (Z_grid.shape[1] - 1)  # horizontal shift (cols)

    lag_X, lag_Y = np.meshgrid(lag_y, lag_x) # possible xy lag pairs

    # convert to physical units
    lag_X = (box_size[0]/nx) * lag_X

    lag_Y = (box_size[1]/ny) * lag_Y

    r = (lag_X**2 + lag_Y**2)**0.5 # total lag distance


    # calculate the variogram values using fft convolutions:
    gamma = scipy.signal.fftconvolve(M, (M*Z_copy**2)[::-1, ::-1], mode='full') + scipy.signal.fftconvolve((M*Z_copy**2), M[::-1, ::-1], mode='full') - 2*scipy.signal.fftconvolve((M*Z_copy), (M*Z_copy)[::-1, ::-1])


    N = scipy.signal.fftconvolve(M, M[::-1, ::-1], mode='full') # normalisation


    svg_values = scipy.stats.binned_statistic(r.flatten(), gamma.flatten(), statistic=np.nansum, bins=bins, range=(0,d_lim)) # bin by total lag distance



    bin_edges = svg_values.bin_edges # extract bins

    bin_centres = (bin_edges[1:] + bin_edges[:-1])/2 # calculate centre of each bin


    svg_values = svg_values.statistic # extract sums
    N_values =  scipy.stats.binned_statistic(r.flatten(), N.flatten(), statistic=np.nansum, bins=bins, range=(0,d_lim)).statistic # bin normalisation


    svg_values = 0.5*(svg_values/N_values) # compute semivariogram

    return svg_values, bin_centres, N_values

def get_physical_coords_grid(header, metadata):
    '''
    convert px to physical coordinate grid from meta
    '''
    world = WCS(header)
    x = np.arange(header['NAXIS1'])
    y = np.arange(header['NAXIS2'])
    X, Y = np.meshgrid(x, y)
    RA_grid, DEC_grid = world.wcs_pix2world(X, Y, 0)

    # Next, convert RA and DEC to physical pc using the meta dict
    delta_RA_deg  = RA_grid  - metadata['RA']
    delta_DEC_deg = DEC_grid - metadata['DEC']
    PA = np.radians(metadata['PA'])
    i  = np.radians(metadata['i'])
    # 1: Rotate RA, DEC by PA to get y (major axis direction) and x (minor axis direction)
    x_deg = delta_RA_deg*np.cos(PA)  - delta_DEC_deg*np.sin(PA)
    y_deg = delta_DEC_deg*np.cos(PA) + delta_RA_deg*np.sin(PA)
    # 2: Stretch x values to remove inclination effects
    x_deg = x_deg / np.cos(i)
    # 3: Convert units to kpc
    x_rad = np.radians(x_deg)
    y_rad = np.radians(y_deg)
    x_kpc = x_rad * metadata['D'] * 1000
    y_kpc = y_rad * metadata['D'] * 1000

    return x_kpc, y_kpc

def get_physical_box_size(x_kpc, y_kpc):
    '''
    get the size of Z_grid from physical coordinates
    '''
    box_size = np.array([x_kpc.max()- x_kpc.min(),y_kpc.max()- y_kpc.min()])
    return box_size

# change path:
data = fits.open('./NGC1385_SFR.fits')

# metadata
metadata = {'Galaxy_ID': 'NGC1385',
 'RA': 54.368,
 'DEC': -24.5012,
 'D': 22.7,
 'log_M_star': 10.22,
 'log_SFR': 0.49,
 'D_SFMS': 0.5,
 'R25_kpc': 8.5,
 'PSF': 0.49,
 'Morphology': 'Sc',
 'PA': 181.3,
 'i': 44.0}

# header
header = data[0].header

# residuals
resid_grid = gg.generate_residual_Z_grid(data[0].data, data[1].data, data[0].header, metadata)


# physical coords
x_kpc, y_kpc = get_physical_coords_grid(header, metadata)

# get the box size
box_size = get_physical_box_size(x_kpc, y_kpc)


# semivariogram
svg, seps, counts = semivariogram_fft(resid_grid, box_size, bins =100, d_lim=2.5)