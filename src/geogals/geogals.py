'''
A collection of functions built for the geostatistical analysis of galaxy data.

Created by: Benjamin Metha, Tree Smith, Jaime Blackwell
Last Updated: May 12, 2025
'''

from . import __version__

import numpy as np
from   sklearn.metrics.pairwise import euclidean_distances
from   astropy.wcs import WCS
import scipy
import emcee
from   statsmodels.regression.linear_model import GLS

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

def to_data_dict(header, Z, e_Z):
    '''
    Parameters
    ----------
    header: hdu header file
        Must contain wcs

    Z: np array
        Grid with values of our random field

    e_Z: np array
        Same shape as Z
        Gives uncertainty of Z at each location

    Returns
    -------
    data_dict: dict
        Contains RA, DEC, Z and e_Z for every
    '''
    RA_grid, DEC_grid = make_RA_DEC_grid(header)
    # Trim off nans
    wanted_pixels = ~np.isnan(Z.flatten()) & ~np.isinf(Z.flatten())
    data_dict = {
        'RA':   RA_grid.flatten()[wanted_pixels],
        'DEC':  DEC_grid.flatten()[wanted_pixels],
        'Z':    Z.flatten()[wanted_pixels],
        'e_Z':  e_Z.flatten()[wanted_pixels]
    }
    return data_dict

#########################
#   Spatial Statistics  #
#########################

def fast_semivariogram(Z_grid, header=None, meta=None, bin_size=2, f_to_keep=1.0):
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

def build_correlated_error_covariance_matrix(dist_matrix, e_Z, meta):
    '''
    Build the covariance matrix due to correlated error associated with the
    measurement of emission lines.
    Assumes PSF of the telescope is a Gaussian.

    Parameters
    ----------

    dist_matrix: (N,N) np.array
        Distances between all pairs of regions.

    e_Z: (N,) np.array
        Uncertainty in metallicity for each observation

    meta: dict
        Metadata used to calculate correlations between.
        Must contain:
        D: float
            Distance from this galaxy to Earth, Mpc.
        PSF: float
            Given in Arcseconds, this is the mean seeing for each galaxy (Mean
            value of Table 1 of Emsellem+22 for native resolution for each galaxy:
            https://ui.adsabs.harvard.edu/abs/2022A%26A...659A.191E/abstract)

    Returns
    -------
    cov_matrix: (N,N) np.array
        Covariance matrix for correlated observation errors.
    '''
    # Convert seeing of 0.6'' to kpc, using small angle approximation
    physical_seeing = meta['PSF']*meta['Dist']*1000/ASEC_PER_RAD
    # Convert seeing (FWHM) into a s.d.
    seeing_sd = physical_seeing / (2*np.sqrt(2*np.log(2))) # from
    # Assume the telescope has a Gaussian PSF:
    correlation_matrix = np.exp(-0.5* (dist_matrix/seeing_sd)**2)
    sd_matrix  = np.diag(e_Z)
    cov_matrix = sd_matrix @ correlation_matrix @ sd_matrix
    return cov_matrix

#########################
#     Model Fitting     #
#########################

def fit_radial_linear_trend(Z_grid, e_Z_grid, header, meta, return_covariances=False):
    '''
    Fits a radial trend to the galaxy data.
    Designed for computing metallicity gradients -- other mean models may be
    required for other galaxy data (e.g. velocities)
    Does not account for small scale variations in the data

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

    optional -- covariance (array)
        Covariance matrix for returned parameters
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
def fit_exp_cov_model(data_dict, meta, n_samples, n_walkers, backend_f, init_theta, init_unc_theta):
    '''
    Fit a geostatistical model to the supplied data using emcee,
    accounting for (1) a radially linear mean trend, and (2) small fluctuations
    that are exponentially correlated

    Parameters
    ----------
    data_dict: dict
        Contains RA, DEC, Z, e_Z for each measured value of our random field.

    meta: dict
        Metadata for this galaxy. Must contain PA (position angle, degrees);
        i (inclination, degrees) and D (distance, Mpc) for this galaxy, as well
        as its central RA and DEC.

    n_samples: int
        Hyperparameter for emcee; controls how many samples are drawn for each
        walker

    n_walkers: int
        Hyperparameter for emcee; controls how many walkers are used to sample
        from the posterior.

    backend_f: str
        Filename for where to store emcee results (.hdf5)

    init_theta: (4,) tuple
        Initial values assumed for log_variance, correlation scale, central
        value of random field, and radial gradient of random field.

    init_unc_theta: (4,) tuple
        Initial uncertainties assumed for the sample parameters

    Returns
    -------
    f_acc: float
        mean acceptance fraction over all chains.
    '''
    r = RA_DEC_to_radius(data_dict['RA'], data_dict['DEC'], meta)
    dist_matrix = deprojected_distances(data_dict['RA'], data_dict['DEC'], meta)
    if 'PSF' in meta:
        # Compute the observational error covariance matrix, accounting for
        # correlation between pixels due to PSF smearing.
        e_Z = build_correlated_error_covariance_matrix(dist_matrix, e_Z=data_dict['e_Z'], meta=meta)
    else:
        # assume all observational error is uncorrelated
        e_Z = np.diag(data_dict['e_Z']**2)
    Z = data_dict['Z']
    # Convert variables that do not change between emcee loops to global variables
    globalize_data(Z, e_Z, r, dist_matrix, init_theta, init_unc_theta)
    # Create arrays of initial values
    pos = np.outer(init_theta, np.ones(n_walkers)) + np.outer(init_unc_theta, np.random.randn(n_walkers))
    pos = pos.T
    backend = emcee.backends.HDFBackend(backend_f)
    # comment out the line below if you want to append results to an existing run.
    backend.reset(n_walkers, 4)
    sampler = emcee.EnsembleSampler(N_WALKERS, N_DIM, exp_4D_log_prob_global, backend=backend)
    sampler.run_mcmc(pos, N_SAMPLES, progress=True, store=True)
    return np.mean(sampler.acceptance_fraction)

def globalize_data(loc_Z, loc_e_Z, loc_r, loc_dist_matrix, loc_init_theta, loc_init_unc_theta):
    '''
    Turn local variables into global ones
    (not strictly necessary, but better for transparency)
    '''
    global Z
    Z = loc_Z
    global e_Z
    e_Z = loc_e_Z
    global r
    r = loc_r
    global dist_matrix
    dist_matrix = loc_dist_matrix
    global init_theta
    init_theta = loc_init_theta
    global init_unc_theta
    init_unc_theta = loc_init_unc_theta
    global n
    n = len(Z)
    global D
    D = np.array([np.ones(n), r]).T
    return 0

def log_likelihood_exp_model(theta):
    '''
    Function that is optimised by emcee to find the best parameters for our
    model of the random field, with radially linear mean values for Z, and
    exponentially-correlated random effects.

    The following must be defined as global variables:

    Z: (N,) np.array
        Observations at N data points

    r: (N,) np.array
        Covariate that is each spaxel's distance from the galaxy center.

    e_Z: (N,N) np.array
        matrix of variance from observational error at all data points.

    dist_matrix: (N,N) np.array
        matrix of distance between all observed data points.

    Parameters
    ----------
    theta: 4-tuple
        Contains:
            log_A, phi: model parameters for spatial_cov
            Z_c, gradZ: model parameters for the large scale gradient

    Returns
    -------
    log_likelihood: float
        The log likelihood of this model with the supplied parameters.
    '''
    log_A, phi, Z_c, gradZ = theta
    A = 10**log_A
    # infold priors
    ln_prior = log_prior(theta, init_theta, init_unc_theta)
    if np.isinf(ln_prior):
        return ln_prior
    r_on_phi  = dist_matrix / phi
    spatial_cov = A* np.exp( -1.0 * r_on_phi)
    V = e_Z + spatial_cov
    try:
        L = np.linalg.cholesky(V)
    except:
        print('Linalg error. Parameter vals:' )
        print(theta)
        return -np.inf
    log_det_L = np.sum(np.log(np.diag(L)))
    log_det_V = 2*log_det_L
    white_D = np.linalg.solve(L, D)
    white_Z = np.linalg.solve(L, Z)
    # subtract mean trend
    beta = np.array([Z_c, gradZ])
    resids =  Z - D @ beta
    white_resids = np.linalg.solve(L, resids)
    chi_sq = n*np.log(2*np.pi) + log_det_V + white_resids.T @ white_resids
    return -0.5*chi_sq + ln_prior

def log_prior(theta, init_theta, init_unc_theta):
    '''
    A prior for the values of the model.
    '''
    # Unpack theta
    log_A, phi, Z_char, gradZ = theta
    init_log_A, init_phi, init_Z_c, init_gradZ = init_theta
    _, _, unc_Z_c, unc_gradZ = init_unc_theta
    # Normal priors on Z_c, grad_Z
    log_Z_c_prior   = log_normal_prior(Z_c,   mu= init_Z_c,  sigma=unc_Z_c)
    log_gradZ_prior = log_normal_prior(gradZ, mu=init_gradZ, sigma=unc_gradZ)
    # Gamma priors on phi, A
    log_A_prior   = log_gamma_prior_tenth_to_ten(10**(log_A - init_log_A))
    log_phi_prior = log_gamma_prior_20_to_500(phi/init_phi)
    return log_Z_char_prior + log_gradZ_prior + log_A_prior + log_phi_prior

def log_normal_prior(x, mu, sigma):
    return -0.5*( ((x-mu)/sigma)**2 ) - np.log(sigma) - 0.5*np.log(2*np.pi)

def log_gamma_prior_tenth_to_ten(x):
    '''Gamma distribution with 1% probability of being below 0.1 or above 10'''
    if x<0:
        return  -np.inf
    alpha = 1.494
    labda = 0.5661
    prob  = np.power(labda, alpha) * np.power(x, alpha - 1) * np.exp(-1.0*labda*x) / gamma(alpha)
    return np.log(prob)

########################################
#   Subsampling and cross-validation   #
########################################

def get_subsample(n_dp, n_in_subsample):
    '''
    Creates an n_dp long array,
    n_in_subsample of which are True,
    the rest of which are False.
    '''
    if n_dp < n_in_subsample:
        raise ValueError
    A = np.zeros(n_dp)
    A[:n_in_subsample] = 1
    np.random.shuffle(A)
    return A == 1

def assign_IDs(n_dp, n_folds):
    '''
    Creates an array of length n_dp, with each element having a random integer
    from 1 to n_folds, and an equal amount of each number.

    If we can't get exactly even groups, the higher numbered groups will have
    one less element than the lower numbers.

    e.g. assign_IDs(5,3) may return [2,3,2,1,1].

    Parameters
    ----------
    n_dp: int
        number of data points

    n_folds: int
        number of groups to split the data into

    Returns
    -------
    group_IDs: np array
        gives ID of each group element.
    '''
    IDs = [x for x in range(1, n_folds+1)]
    long_IDs = IDs * int(n_dp/n_folds + 1)
    group_IDs = np.array(long_IDs[:n_dp])
    np.random.shuffle(group_IDs)
    return group_IDs

#########################
#        Kriging        #
#########################

def krig_exp_model(RA, DEC, Hii_df, meta, theta, mode='grid'):
    ##### NOTE! Only tested for grid so far.
    '''
    Performs universal kriging on a model grid of RA and DEC
    Uses my distance function, a choice of covariance function, the best fitting
    value of f_d (no restrictions), and assumes Z ~ r + (random effects)

    Uses equations presented in 'Spatio-Temporal Statistics with R', available
    for free at https://spacetimewithr.org

    Parameters
    ----------
    RA: np array
        Array of RA values. Must be in degrees.

    DEC: (N,) np array
        Array of DEC values. Must be in degrees.

    Z_df: data dict, containing RA, DEC, Z, e_Z for each data point.

    meta: dict
        Metadata used to calculate the distances. Must contain:
        PA: float
            Principle Angle of the galaxy, degrees.
        i: float
            inclination of the galaxy along this principle axis, degrees.
        D: float
            Distance from this galaxy to Earth, Mpc.

    theta: (4,) tuple
        Contains model parameters (log_Var, phi, Zc, gradZ)

    mode: str
        Options include
        'grid': make a grid of all possible combos of supplied RA and DEC
                values; use kriging to estimate Z at each point on grid.
        'list': just use pairs of RA and DEC values as they are given.
                RA and DEC must have the same length.
        'auto': Get RA and DEC values from the df itself.

    Returns
    -------
    Z_pred_matrix: (M,N) np array
        interpolated (kriged) values over the RA, DEC coords given.

    var_matrix: (M,N) np array
        variances for these predictions
    '''
    log_A, phi, Z_c, gradZ = theta
    A = 10**log_A
    if mode=='grid':
        # Construct arrays for all pairs of RA and DEC values
        if RA.shape == DEC.shape:
            shape = RA.shape
        else:
            raise ValueError("RA_grid and DEC_grid must have the same shape! Input RA shape: {0} Input DEC shape: {1}".format((RA_grid.shape, DEC_grid.shape)))
        RA_long = RA.flatten()
        DEC_long = DEC.flatten()
    elif mode=='list':
        RA_long = RA
        DEC_long = DEC
    elif mode=='auto':
        RA_long = df['RAdeg']
        DEC_long = df['DEdeg']
    else:
        print("Error: Bad argument given to `krig_model`.\nMode must be either 'grid', 'list', or 'auto'.")
        return np.nan, np.nan

    data_dists = deprojected_distances(Z_df['RA'], Z_df['DEC'], meta)
    r = RA_DEC_to_radius(Z_df['RA'], Z_df['DEC'], meta)
    Z_resids = Z_df['Z'] - Z_c - gradZ*r
    data_grid_dists = deprojected_distances(Z_df['RA'], Z_df['DEC'], RA_long, DEC_long, meta=meta)
    # Find galactocentric radius of each grid point and each data point
    grid_r = deprojected_distances(RA_long, DEC_long, meta['RA'], meta['DEC'], meta=meta).T[0]
    covariates = np.array([np.ones(len(r)), r]).T
    grid_covariates = np.array([np.ones(len(grid_r)), grid_r]).T
    best_beta = np.array([Z_c, gradZ])
    if 'PSF' in meta:
        # Compute the observational error covariance matrix, accounting for
        # correlation between pixels due to PSF smearing.
        error_cov = build_correlated_error_covariance_matrix(dist_matrix, e_Z=data_dict['e_Z'], meta=meta)
    else:
        # assume all observational error is uncorrelated
        error_cov = np.diag(data_dict['e_Z']**2)

    spatial_cov_data = A*np.exp(-1.0*data_dists/phi)
    spatial_cov_data_grid = A*np.exp(-1.0*data_grid_dists/phi)
    tot_data_cov = spatial_cov_data + error_cov

    # Use Universal Kriging to estimate residuals for each grid point
    # SpaceTimeWithR, equation 4.6
    c_factor = scipy.linalg.cho_factor(tot_data_cov)
    white_resids = scipy.linalg.cho_solve(c_factor, Z_resids)
    predicted_grid_resids = spatial_cov_data_grid.T @ white_resids
    # Add this to model to predict metallicity at each point.
    predicted_grid_Z = np.dot(best_beta, grid_covariates.T) + predicted_grid_resids

    # Get uncertainty (eq. 4.10 of SpaceTimeWithR)
    white_D = scipy.linalg.cho_solve(c_factor, covariates)
    white_cov_data_grid = scipy.linalg.cho_solve(c_factor, spatial_cov_data_grid)
    Cov_cvars = covariates.T @ white_D
    inv_cov_cvars =np.linalg.inv(Cov_cvars) # don't be fancy, since it's a 2 by 2
    kriged_cvar_resids = grid_covariates - (spatial_cov_data_grid.T @ white_D)
    gls_uncertainty = np.diagonal( kriged_cvar_resids @ inv_cov_cvars @ kriged_cvar_resids.T)
    grid_uncertainty = A - np.diagonal(spatial_cov_data_grid.T @ white_cov_data_grid) + gls_uncertainty

    # If needed, reshape these to be an RA X DEC shaped, plottable matrix
    if mode=='grid':
        Z_pred_matrix = predicted_grid_Z.reshape(shape)
        var_matrix = grid_uncertainty.reshape(shape)
        return Z_pred_matrix, var_matrix
    else:
        return predicted_grid_Z, grid_uncertainty
