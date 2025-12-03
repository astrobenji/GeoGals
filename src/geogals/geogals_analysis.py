from geogals_io import *
from geogals_log import *

from meta import *

import numpy as np
import matplotlib.pyplot as plt
import copy
from statsmodels.regression.linear_model import GLS
import inspect
import scipy

class GeoMap(SimulationMeta):
    """
    Container for two-dimensional maps derived from simulation or observational
    data. Supports fitting one-dimensional radial models, storing model
    parameters, computing residual maps, generating semivariograms, and plotting
    profiles and residuals.

    Notes
    -----
    This class inherits from ``SimulationMeta`` and therefore also provides
    access to metadata handling, logging, file management, and stored simulation
    properties.
    """

    def __init__(self, galaxy_name, results_base_directory='./results',
                 meta=None, run_id=None, **parameters):
        """
        Initialize a GeoMap instance and load associated metadata.

        Parameters
        ----------
        galaxy_name : str
            Name of the galaxy or simulation run.
        results_base_directory : str, optional
            Base directory where results are written. Default is ``'./results'``.
        meta : Meta-like object or dict, optional
            Metadata object or dictionary used to initialize the underlying
            ``SimulationMeta`` information. If ``None``, metadata must be loaded
            using ``run_id``.
        run_id : int or str, optional
            Identifier for a simulation run. Used when ``meta`` is not supplied.
        **parameters : dict
            Additional run parameters forwarded to ``load_meta``.

        Notes
        -----
        Sets the internal filename tag to ``'map'``.
        """
        super().__init__(galaxy_name, results_base_directory)
        self.load_meta(meta, run_id, load_into='map', **parameters)
        self._fname = 'map'
    
    # def _create_subrun(self, **run_params):
    #     self._create_run(parameters=run_params, run_id=self.run_id)


    def _property(self, property_name):
        """
        Retrieve a stored map property by name, optionally applying NaN and
        radial masks.

        Parameters
        ----------
        property_name : str
            Name of the attribute containing the property values.
        nan_mask : bool, optional
            If ``True``, mask out NaN and infinite values. Default is ``True``.
        max_r : float or str, optional
            Maximum radial extent. If a string, interpreted as a header key.
            If ``None``, no radial mask is applied. Default is ``None``.
        return_r : bool, optional
            If ``True``, also return the radial distances corresponding to the
            returned property values. Default is ``False``.

        Returns
        -------
        ndarray or tuple of ndarray
            The masked property values, and optionally the corresponding radii.

        Raises
        ------
        AttributeError
            If no attribute with the given property name exists.

        Notes
        -----
        - If both ``max_r`` and ``nan_mask`` are supplied, both masks are
          applied.
        - ``self.R`` must contain the per-pixel radial distance array.
        """
        if hasattr(self, property_name):
            vals = getattr(self, property_name)
        else:
            raise AttributeError(f'no property named {property_name}')
        return vals



        
    def _get_property(self, property_name, nan_mask=True, max_r=None, return_r=False, return_sigma = False):
        vals = self._property(property_name)
        pixels_to_keep = None
        if max_r is not None:
            pixels_to_keep = self._r_mask(max_r)
            if nan_mask:
                pixels_to_keep *= self._nan_mask(vals)
        elif nan_mask:
            pixels_to_keep = self._nan_mask(vals)
        
        vals = self._apply_mask(vals, mask=pixels_to_keep)
        if return_sigma:
            sigma = self._e_property(property_name)
            sigma = self._apply_mask(sigma, mask=pixels_to_keep)
            if return_r:
                
                return vals,self._apply_mask(self.R, mask=pixels_to_keep), sigma
            return vals, sigma
        
        if return_r:
            return vals,self._apply_mask(self.R, mask=pixels_to_keep)
        return vals
        
    def _e_property(self, property_name):
        if hasattr(self, f'e_{property_name}'):
            return getattr(self, f'e_{property_name}')
        else:
            return None
        
    def _apply_mask(self, values, mask):
        if values is None or mask is None:
            return values

        
        return values[mask]

    def _header_value(self, header_property):
        """
        Retrieve a value from stored header metadata or as an attribute.

        Parameters
        ----------
        header_property : str
            Name of the header field to retrieve.

        Returns
        -------
        object
            Value stored in either an attribute or ``self.data_header``.

        Raises
        ------
        AttributeError
            If the property cannot be found in either location.
        """
        if hasattr(self, header_property):
            return getattr(self, header_property)
        else:
            if hasattr(self, 'data_header') and header_property in self.data_header:
                return self.data_header[header_property]
            else:
                raise AttributeError(f'no value for {header_property}')

    def _get_radial_maximum(self, max_r):
        """
        Convert a radial limit specification to a numeric value.

        Parameters
        ----------
        max_r : float, int, str, or None
            If a string, interpreted as a header field. If ``None``, returns
            ``np.inf``.

        Returns
        -------
        float
            Maximum radius value.

        Raises
        ------
        TypeError
            If the resolved ``max_r`` is of an unsupported type.
        """
        if max_r is not None:
            if isinstance(max_r, str):
                max_r = self._header_value(max_r)

            if isinstance(max_r, float) or isinstance(max_r, int):
                pass
            else:
                raise TypeError(f'maximum r value is {type(max_r)}')
        else:
            max_r = np.inf

        return max_r

    def _nan_mask(self, values):
        """
        Mask NaN and infinite entries in an array.

        Parameters
        ----------
        values : ndarray
            Array to mask.

        Returns
        -------
        ndarray of bool
            Boolean mask of valid values.
        """
        return ~np.isnan(values) & ~np.isinf(values)

    def _r_mask(self, max_r):
        """
        Mask values whose radial distance exceeds a specified maximum.

        Parameters
        ----------
        max_r : float or int
            Radial cutoff.

        Returns
        -------
        ndarray of bool
            Boolean mask where ``self.R < max_r``.
        """
        return self.R < max_r



    def _get_polynomial(self, property_name, max_r, degree, recompute=False, override=False):
        """
        Retrieve or compute the fitted radial gradient for a property.

        Parameters
        ----------
        property_name : str
            Name of the mapped property.
        max_r : float or str
            Maximum radius used in the gradient fit.
        recompute : bool, optional
            If ``True``, forces recomputation even if a gradient dictionary
            exists. Default is ``False``.
        override : bool, optional
            Passed through to metadata comparison routines. Default is ``False``.

        Returns
        -------
        dict
            Dictionary containing parameters, covariance, residuals, and
            ``max_r``.
        """
        if degree ==1:
            dict_name = f'{property_name}_gradient_dict'
        else:
            dict_name = f'{property_name}_polynomial_{degree}_dict'
        if not hasattr(self, dict_name) or recompute:
            polynomial_dict = self._fit_polynomial(property_name, max_r, degree, override)
        else:
            polynomial_dict = getattr(self, dict_name)
            if polynomial_dict['max_r'] != max_r:
                polynomial_dict = self._get_polynomial(
                    property_name, max_r, degree,
                    recompute=True,
                    override=override
                )
        return polynomial_dict
    
    def _fit_polynomial(self, property_name, max_r, degree, override=False):
        """
        Fit a polynomial of specified degree to radial property values.

        Parameters
        ----------
        property_name : str
            Name of the property to fit.
        max_r : float or str
            Maximum radius for the fit.
        degree : int
            Polynomial degree.

        Returns
        -------

        """
        max_r_val = self._get_radial_maximum(max_r)
        values, r_values, sigma = self._get_property(property_name, nan_mask=True,
                                          max_r=max_r_val, return_r=True, return_sigma = True)
        cov = np.vander(r_values, degree + 1, increasing=True)
        poly_model = GLS(values, cov, sigma).fit()
        params = poly_model.params
        cov = poly_model.normalized_cov_params

        polynomial_dict = {'params': params, 'cov': cov, 'max_r': max_r, 'degree': degree}
        residuals = self._polynomial_residuals(
            property_name, params, max_r_val
        )
        polynomial_dict['residuals'] = self._polynomial_residuals(
            property_name, params, max_r_val
        )

        if degree == 1:
            self._compare_attribute_with_argument(
                **{f'{property_name}_gradient_dict': polynomial_dict},
                override=override
            )
        else:
            self._compare_attribute_with_argument(
                **{f'{property_name}_polynomial_{degree}_dict': polynomial_dict},
                override=override
            )
        return polynomial_dict
    
    def _polynomial_residuals(self, property_name, params, max_r):
        """
        Compute per-pixel residuals from a fitted polynomial model.

        Parameters
        ----------
        property_name : str
            Name of the property.
        params : array_like
            Fitted gradient parameters ``[, ]``.
        max_r : float
            Maximum radius for applying the model.

        Returns
        -------
        ndarray
            Residual map of ``property - model``.
        """

        degree = len(params) - 1

        params = params[::-1]

        mu_grid = np.sum([params[i] * self.R**(degree-i) for i in range(len(params))], axis=0)
        mu_grid[~self._r_mask(max_r)] = np.nan


        property_grid = self._get_property(property_name, nan_mask=False,
                                       max_r=None, return_sigma=False)

        return property_grid - mu_grid
    
    
        

  
    def _scatter_r_against_property(self, property_name, max_r_val,
                       point_colour='tab:blue',
                       point_size=0.1,
                       ):
        values, r_values = self._get_property(property_name, nan_mask=True,
                                          max_r=max_r_val, return_r=True)
        
        plt.scatter(r_values, values, c=point_colour, s=point_size)
        
        
    
    def _plot_polynomial(self, property_name, params, max_r,
                       point_colour='tab:blue', line_colour='k',
                       point_size=0.1, save_plot=False, fname=None,
                       **kwargs):
        """
        Plot radial property values with a fitted polunomial line.

        Parameters
        ----------
        property_name : str
            Name of the property to plot.
        parameters : array_like
            Gradient coefficients: intercept and slope.
        max_r : float or str
            Maximum radius.
        point_colour : str, optional
            Matplotlib scatter color for data points.
        line_colour : str, optional
            Matplotlib color for the fitted line.
        point_size : float, optional
            Marker size. Default is ``0.1``.
        save_plot : bool, optional
            If ``True``, save the plot. Default is ``False``.
        fname : str, optional
            Filename for saving. If ``None``, a default is constructed.
        **kwargs : dict
            Passed to labeling utilities and ``plt.xlabel``.

        Notes
        -----
        Saves using ``self.file_handler.save_plot`` when requested.
        """
        max_r_val = self._get_radial_maximum(max_r)
        r = np.linspace(0, max_r_val)
        self._scatter_r_against_property(property_name, max_r_val=max_r_val, point_colour=point_colour, point_size=point_size)
        degree = len(params)-1
        params = params[::-1]
        line_vals = np.sum([params[i] * r**(degree-i) for i in range(len(params))], axis=0)

        
        plt.plot(r, line_vals, c=line_colour)
        self._label_radial_plot(property_name, **kwargs)

        if save_plot:
            fname = self._get_gradient_plot_filename(property_name, fname)
            self.file_handler.save_plot(fname, self.run_id)
            self.logger.log_save(self.run_id, fname, 'png')

    def polynomial(self, property_name, max_r,degree, plot=False,
                 override=False, **kwargs):
        """
        Fit or retrieve the stored radial gradient for a property.

        Parameters
        ----------
        property_name : str
            Name of the property.
        max_r : float or str
            Maximum radius for the fit.
        plot : bool, optional
            If ``True``, plot the fitted gradient. Default is ``False``.
        recompute : bool, optional
            If ``True``, recompute fit even if stored. Default is ``True``.
        override : bool, optional
            Passed to metadata comparison routines. Default is ``False``.
        **kwargs : dict
            Additional parameters passed to plotting routines.

        Returns
        -------
        params : ndarray
            Fitted gradient parameters (intercept, slope).
        cov : ndarray
            Covariance matrix of the fit.
        """
        polynomial_dict = self._get_polynomial(
            property_name, max_r, degree, override
        )
        params = polynomial_dict['params']
        cov = polynomial_dict['cov']

        if plot:
            self._plot_polynomial(property_name, params, max_r, **kwargs)

        return params, cov

    def gradient(self, property_name, max_r,degree, plot=False,
                 override=False, **kwargs):
        """
        Fit or retrieve the stored radial gradient for a property.

        Parameters
        ----------
        property_name : str
            Name of the property.
        max_r : float or str
            Maximum radius for the fit.
        plot : bool, optional
            If ``True``, plot the fitted gradient. Default is ``False``.
        recompute : bool, optional
            If ``True``, recompute fit even if stored. Default is ``True``.
        override : bool, optional
            Passed to metadata comparison routines. Default is ``False``.
        **kwargs : dict
            Additional parameters passed to plotting routines.

        Returns
        -------
        params : ndarray
            Fitted gradient parameters (intercept, slope).
        cov : ndarray
            Covariance matrix of the fit.
        """
        params, cov = self.polynomial(self, property_name, max_r,degree, plot,
                 override, **kwargs)

        return params, cov


    def _label_radial_plot(self, property_name, **kwargs):
        """
        Label the gradient plot axes using header information when possible.

        Parameters
        ----------
        property_name : str
            Name of the property.
        **kwargs : dict
            Additional keyword arguments passed to ``plt.xlabel``.
        """
        sig = inspect.signature(plt.xlabel)
        labels_dict = {
            key: val for key, val in kwargs.items()
            if key in sig.parameters.keys()
        }

        x_label = self._get_labels('distance', 'arbitrary units',
                                   label='r', **kwargs)
        plt.xlabel(x_label, **labels_dict)

        y_label = self._get_labels(property_name, 'unknown units', **kwargs)
        plt.ylabel(y_label, **labels_dict)

    def _get_labels(self, property_type, default_units, **kwargs):
        """
        Construct axis labels for plotting.

        Parameters
        ----------
        property_type : str
            Type of property (e.g., ``'distance'`` or the property name).
        default_units : str
            Units to fall back to when none are available.
        **kwargs : dict
            May include:
            - ``label`` : override the label text
            - ``{property_type}_units`` : override units

        Returns
        -------
        str
            Formatted label including units.
        """
        if f'{property_type}_units' in kwargs:
            property_units = kwargs[f'{property_type}_units']
        else:
            try:
                property_units = self._header_value(
                    f'{property_type}_units'
                )
            except Exception:
                property_units = default_units

        if 'label' in kwargs:
            property_label = kwargs['label']
        else:
            property_label = property_type

        return f'{property_label} ({property_units})'



    def _get_gradient_residuals(self, property_name, max_r):
        """
        Retrieve stored residuals for a gradient fit.

        Parameters
        ----------
        property_name : str
            Name of the property.
        max_r : float or str
            Maximum radius used for the gradient.

        Returns
        -------
        ndarray
            Residual field corresponding to the gradient model.
        """
        gradient_dict = self._get_polynomial(property_name, max_r, degree=1)
  
        return gradient_dict['residuals']

    def _semivariogram(self, property_name, max_r, bins=50,
                       separation_cutoff=None, return_bin_centres=True,
                       mu_model='gradient'):
        """
        Compute the semivariogram of a residual map.

        Parameters
        ----------
        property_name : str
            Name of the property whose residuals are used.
        max_r : float or str
            Maximum radius for the residual computation.
        bins : int or array-like, optional
            If int, number of equal-width radial bins. If array-like,
            treated as explicit bin edges.
        separation_cutoff : float, optional
            Maximum separation included in the computation. If ``None``,
            uses the maximum available. Default is ``None``.
        return_bin_centres : bool, optional
            If ``True``, return bin centres instead of edges. Default is ``True``.
        mu_model : str, optional
            Currently unused, placeholder for future extensions. Default is
            ``'gradient'``.

        Returns
        -------
        bin_centres or bin_edges : ndarray
            Separation distance corresponding to each semivariogram value.
        semivariogram_values : ndarray
            Semivariogram values in each bin.
        counts : ndarray
            Number of point pairs contributing to each bin.

        Notes
        -----
        - Works on a 2D grid of residual values.
        - Uses FFT-based convolution for efficient computation.
        - ``self.box_size`` must define the physical size of the map.
        """

        box_size = getattr(self, 'box_size')
        residuals = self._get_gradient_residuals(property_name, max_r)
        data_grid = residuals.copy()

        nx, ny = data_grid.shape
        pad_shape = (2 * nx - 1, 2 * ny - 1)

        M = np.isnan(data_grid)
        data_copy = np.zeros_like(data_grid)
        data_copy[~M] = data_grid[~M]
        M = (~M).astype(float)

        lag_y = np.arange(pad_shape[0]) - (data_grid.shape[0] - 1)
        lag_x = np.arange(pad_shape[1]) - (data_grid.shape[1] - 1)

        lag_Y, lag_X = np.meshgrid(lag_x, lag_y)

        lag_X = (box_size[0] / nx) * lag_X
        lag_Y = (box_size[1] / ny) * lag_Y

        r = np.sqrt(lag_X**2 + lag_Y**2)

        gamma = (scipy.signal.fftconvolve(
                     M, (M * data_copy**2)[::-1, ::-1], mode='full')
                 + scipy.signal.fftconvolve(
                     (M * data_copy**2), M[::-1, ::-1], mode='full')
                 - 2 * scipy.signal.fftconvolve(
                     (M * data_copy), (M * data_copy)[::-1, ::-1]))

        N = scipy.signal.fftconvolve(M, M[::-1, ::-1], mode='full')

        bins = np.array([bins]).flatten()
        if len(bins) > 1:
            bin_edges = bins
        else:
            bins = bins[0]
            if separation_cutoff is None:
                separation_cutoff = r.max()
            bin_edges = np.linspace(0, separation_cutoff, bins + 1)

        semivariogram_info = scipy.stats.binned_statistic(
            r.flatten(), gamma.flatten(),
            statistic=np.nansum, bins=bin_edges
        ).statistic

        counts = scipy.stats.binned_statistic(
            r.flatten(), N.flatten(),
            statistic=np.nansum, bins=bin_edges
        ).statistic

        semivariogram_values = 0.5 * (semivariogram_info / counts)

        if return_bin_centres:
            bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            return bin_centres, semivariogram_values, counts
        else:
            return bin_edges, semivariogram_values, counts
