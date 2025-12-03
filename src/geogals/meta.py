import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from geogals_io import *
from geogals_log import *

"""
By Tree

Module for storing metadata and physical coordinate information.
Provides abstract base metadata handling for simulation metadata (IFU data meta still to come!), and
utilities for interacting with the filesystem.

"""

import pickle
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from geogals_io import *
from geogals_log import *


class Meta(ABC):
    """
    Abstract base class for managing metadata of simulation or observational data.

    This class provides common utilities for saving, loading, and handling metadata.
    Subclasses must implement methods for constructing from dictionaries and
    exposing a `parameters` property describing run-defining parameters.

    Parameters
    ----------
    galaxy_name : str
        Name of the galaxy used to derive file paths.
    results_base_directory : str, optional
        Base directory for storing results. Default is './results/'.

    Attributes
    ----------
    _fname : str
        Base filename for saving/loading metadata objects. Defaults to 'meta'.
    galaxy_name : str
        Name of the galaxy.
    file_handler : FileHandler
        Object for handling filesystem operations.
    logger : Logger
        Logger instance for recording metadata events.
    """

    def __init__(self, galaxy_name, results_base_directory='./results/'):
        """
        Initialize a Meta object with file handler and logger.

        Parameters
        ----------
        galaxy_name : str
            Name of the galaxy.
        results_base_directory : str, optional
            Root directory for storing results.
        """
        self._fname = 'meta'
        self.galaxy_name = galaxy_name
        self.file_handler = FileHandler(galaxy_name, results_base_directory)
        self.logger = Logger(galaxy_name, results_base_directory)

    @abstractmethod
    def read_meta_from_dictionary(self, metadict):
        """
        Populate the metadata object from a dictionary.

        Subclasses must override this method to parse specific metadata
        fields and populate instance attributes.

        Parameters
        ----------
        metadict : dict
            Dictionary containing metadata fields.

        Returns
        -------
        self : Meta
            Populated metadata object.
        """
        pass

    @property
    @abstractmethod
    def parameters(self):
        """
        Return a dictionary of run-defining parameters.

        Subclasses must implement this property to provide a stable mapping
        that uniquely identifies a run.

        Returns
        -------
        dict
            Dictionary of run-defining parameters.
        """
        pass

    def read_meta_from_keywords(self, **kwargs):
        """
        Convenience method to populate metadata from keyword arguments.

        Returns
        -------
        self : Meta
            Populated metadata object.
        """
        return self.read_meta_from_dictionary(kwargs)

    def save_to_pickle(self):
        """
        Persist the metadata object to a pickle file in its run directory.

        Notes
        -----
        Uses the `FileHandler.pickle_object` method to save the object.
        """
        self.file_handler.pickle_object(self)

    @abstractmethod
    def load_from_pickle(self, run_id=None, parameters=None):
        """
        Load a metadata object from pickle storage.

        Subclasses must accept either an explicit `run_id` or a set of
        identifying `parameters` and return a metadata instance.

        Parameters
        ----------
        run_id : str, optional
            Explicit run identifier to load.
        parameters : dict, optional
            Parameter mapping to identify a run if `run_id` is not provided.

        Returns
        -------
        Meta
            Loaded metadata object.
        """
        pass

    def _compare_attribute_with_argument(self, override=False, **kwargs):
        """
        Ensure required attributes exist or set them from keyword arguments.

        Parameters
        ----------
        override : bool, optional
            If True, overwrite existing attributes. Default is False.
        **kwargs
            Attribute names and values to verify or assign.

        Raises
        ------
        AttributeError
            If a required attribute is missing and no value is provided.
        """
        for key, value in kwargs.items():
            if value is None and not hasattr(self, key):
                raise AttributeError(f"No value for '{key}' stored or provided.")
            elif value is not None and (override or not hasattr(self, key)):
                setattr(self, key, value)

            if hasattr(self, 'run_id'):
                self.logger.log_load_attrs(self.run_id, self._fname, key, value)
            else:
                self.logger.log_load_attrs(None, self._fname, key, value)
    
    def _create_run(self, parameters=None, run_id=None):
        """
        Create a run directory or subrun directory with a determenitistic identifier based on `parameters`.

        If a run_id is passed, a subrun directory is created within the directory for that run.

        If no run_id is passed, a run directory is created and the run_id attribute is set


        Returns
        -------

        None
        """
        if run_id is None:
            run_id = self._run_id(parameters)
            setattr(self, 'run_id', run_id)
            subrun_id = None

        else:
            subrun_id = self._run_id(parameters)

    
        if not self.file_handler._get_run_directory(run_id, subrun_id).exists():
            self.file_handler.create_run_directory(run_id, subrun_id)
            self.logger.log_run(run_id)
            self.logger.log_params(run_id, params_dict=self.parameters)

    def _run_id(self,  parameters = None):
        """
        Compute a deterministic run identifier based on `parameters`.

        The identifier is the first 8 characters of the MD5 hash of the JSON-
        serialized parameters dictionary (with stable key ordering).

        Returns
        -------
        run_id : str
            Deterministic run identifier.
        """
        import hashlib, json

        if parameters is None:
            parameters = self.parameters
        s = json.dumps(parameters, sort_keys=True).encode()
        run_id = hashlib.md5(s).hexdigest()[:8]



        return run_id

    def load_meta(self, meta=None, run_id=None, load_into=None, **search_params):
        """
        Load attributes from another Meta instance or from storage.

        Parameters
        ----------
        meta : Meta, optional
            Metadata object to copy attributes from.
        run_id : str, optional
            Explicit run identifier to load.
        load_into : str, optional
            Name of the receiving object (for logging only).
        **search_params : dict
            Parameters to locate a stored metadata object.

        Raises
        ------
        ValueError
            If no metadata object can be loaded.
        """
        if meta is not None:
            self.__dict__.update(meta.__dict__)
            self.logger.log_init(load_into, self.run_id)
            self.logger.log_load(self.run_id, 'meta', load_into=load_into)
        else:
            try:
                meta = self.load_from_pickle(run_id=run_id, parameters=search_params,
                                             log=False, fname='meta')
                self.load_meta(meta=meta, load_into=load_into)
            except:
                raise ValueError('No meta for this galaxy exists')

    def read_header(self, data_path, **kwargs):
        """
        Read metadata from an HDF5 or FITS header.

        Parameters
        ----------
        data_path : str or Path
            Path to the data file.
        store : bool, optional
            If True, store header fields in the metadata object.
        override : bool, optional
            If True, overwrite existing attributes.

        Returns
        -------
        dict
            Header dictionary if store=False.
        """
        store = kwargs.pop('store', False)
        override = kwargs.pop('override', False)
        header = self.file_handler.load_header_from_file(data_path, **kwargs)
        self.logger.log_read(self.run_id, 'header', data_path)

        if store:
            self._store_header_from_dict(header, store, override)
        else:
            return header

    def _store_header_from_dict(self, header, store='as_dict', override=False):
        """
        Store header dictionary as object attributes or as a single dict.

        Parameters
        ----------
        header : dict
            Header dictionary to store.
        store : {'as_dict', 'as_attrs'}, optional
            'as_dict' stores under self.data_header; 'as_attrs' stores each key as attribute.
        override : bool, optional
            If True, overwrite existing attributes.

        Returns
        -------
        dict
            Header dictionary.
        """
        if store:
            if store == 'as_dict':
                self._compare_attribute_with_argument(data_header=header, override=override)
            elif store == 'as_attrs':
                self._compare_attribute_with_argument(**header, override=override)
        return header

class SimulationMeta(Meta):
    """
    Metadata container for regularly sampled simulation grids.

    Stores grid dimensions in three interchangeable forms: `N_px` (number of pixels),
    `box_size` (physical size), and `resolution` (physical size per pixel). At least
    two must be supplied; the third is computed.

    Attributes
    ----------
    N_px : ndarray of int, shape (2,)
        Number of pixels along each axis (x, y).
    box_size : ndarray of float, shape (2,)
        Physical size of the simulation box along each axis.
    resolution : ndarray of float, shape (2,)
        Physical size of a single pixel along each axis.
    boundaries : ndarray, shape (2,2)
        Physical boundaries of the box: [[x_min, x_max], [y_min, y_max]].
    px_bounds : ndarray, shape (2, N_px[i]+1)
        Pixel boundary coordinates for each axis.
    X, Y : ndarray
        Meshgrid arrays of pixel centre coordinates.
    R : ndarray
        Radial distances computed from X and Y.
    """

    def __init__(self, galaxy_name, results_base_directory='./results/'):
        """
        Initialize a SimulationMeta object.

        Parameters
        ----------
        galaxy_name : str
            Name of the galaxy.
        results_base_directory : str, optional
            Base directory for results. Default is './results/'.
        """
        super().__init__(galaxy_name, results_base_directory)

    def read_meta_from_dictionary(self, metadict):
        """
        Populate the object from a metadata dictionary.

        Computes missing quantities among `N_px`, `box_size`, and `resolution`.
        Initializes spatial boundaries, coordinate grids, and run ID.

        Parameters
        ----------
        metadict : dict
            Dictionary containing at least two of `N_px`, `box_size`, `resolution`.

        Returns
        -------
        SimulationMeta
            The populated object.

        Raises
        ------
        KeyError
            If fewer than two primary parameters are provided.
        """
        metadict = self._read_meta_params(**metadict)
        self.logger.log_init('meta')
        self._compare_attribute_with_argument(**metadict)
        self._set_boundaries()
        self._set_coords()
        self._create_run()
        return self

    @property
    def parameters(self):
        """
        Return a dictionary of parameters that uniquely define the run.

        Returns
        -------
        dict
            Keys: 'N_px_x', 'N_px_y', 'resolution_x', 'resolution_y',
                  'box_size_x', 'box_size_y'.
        """
        parameters = {}
        for param in ['N_px', 'resolution', 'box_size']:
            param_value = getattr(self, param)
            parameters[param + '_x'] = param_value[0].item()
            parameters[param + '_y'] = param_value[1].item()
        return parameters

    def load_from_pickle(self, run_id=None, parameters=None, log=True, fname=None):
        """
        Load a SimulationMeta object from pickle storage.

        Parameters
        ----------
        run_id : str, optional
            Explicit run identifier.
        parameters : dict, optional
            Parameter mapping to search for a matching run if run_id not provided.
        log : bool, optional
            Passed to FileHandler.load_pickle to control logging. Default True.
        fname : str, optional
            Filename base to load from. Defaults to self._fname.

        Returns
        -------
        SimulationMeta
            Loaded metadata object.

        Raises
        ------
        ValueError
            If neither `run_id` nor `parameters` are provided.
        """
        if fname is None:
            fname = self._fname
        if run_id is not None:
            return self.file_handler.load_pickle(fname=fname, run_id=run_id, log=log)
        elif parameters is not None:
            run_id = RunIndexer(self.galaxy_name,
                                self.file_handler.results_directory.parent)\
                     .params_to_run(simulation=True, n_matches=2, **parameters)
            return self.load_from_pickle(run_id=run_id, log=log)
        else:
            raise ValueError('One of run_id or parameters must not be None.')

    def _read_meta_params(self, N_px=None, resolution=None, box_size=None, **kwargs):
        """
        Validate and set N_px, resolution, and box_size.

        At least two of the three parameters must be provided. Missing quantities
        are computed deterministically. Sets attributes on the instance.

        Parameters
        ----------
        N_px : int or sequence of int, optional
            Number of pixels along each axis.
        resolution : float or sequence of float, optional
            Physical size per pixel.
        box_size : float or sequence of float, optional
            Physical box size along each axis.
        **kwargs
            Additional metadata fields.

        Returns
        -------
        dict
            The remaining keyword arguments (for API compatibility).

        Raises
        ------
        KeyError
            If fewer than two of the primary inputs are provided.
        """
        if np.sum([var is not None for var in [N_px, resolution, box_size]]) < 2:
            raise KeyError("Input dictionary must contain at least two of N_px, box_size, or resolution.")

        if N_px is not None:
            self._format_parameters(N_px, 'N_px')
        if box_size is not None:
            self._format_parameters(box_size, 'box_size')
        if resolution is not None:
            if box_size is None or N_px is None:
                self._format_parameters(resolution, 'resolution')
            else:
                self._check_resolution_consistency(resolution)
        else:
            resolution = self.resolution_from_box_Npx(self.box_size, self.N_px)
            self._format_parameters(resolution, 'resolution')

        if N_px is None:
            N_px = self.N_px_from_box_res(self.box_size, self.resolution)
            self._format_parameters(N_px, 'N_px')
        if box_size is None:
            box_size = self.box_size_from_N_px_res(self.N_px, self.resolution)
            self._format_parameters(box_size, 'box_size')

        return kwargs

    def _format_parameters(self, param, param_name=None):
        """
        Convert parameter to length-2 numpy array and optionally assign.

        Scalars are duplicated to form a 2-element array. N_px is cast to int.

        Parameters
        ----------
        param : scalar or sequence
            Input parameter value(s).
        param_name : str, optional
            Attribute name to assign the formatted array.

        Returns
        -------
        ndarray
            Formatted array if param_name is None.
        """
        param = np.array([param]).flatten()
        if len(param) == 1:
            param = np.array([param[0], param[0]])
        assert len(param) == 2, "Too many dimensions supplied."

        if param_name is not None:
            if param_name == 'N_px':
                param = np.ceil(param).astype(int)
            setattr(self, param_name, param)
        else:
            return param

    def _check_resolution_consistency(self, input_resolution):
        """
        Check supplied resolution against computed value and update if needed.

        Parameters
        ----------
        input_resolution : scalar or sequence
            User-supplied resolution values.

        Returns
        -------
        None
            Updates self.resolution if inconsistent.
        """
        input_resolution = self._format_parameters(input_resolution)
        calculated_resolution = self.resolution_from_box_Npx(self.box_size, self.N_px)
        if not np.allclose(input_resolution, calculated_resolution):
            print("Warning: Supplied resolution is inconsistent with box_size and N_px. "
                  "Overriding with calculated value.")
        self._format_parameters(calculated_resolution, 'resolution')

    def resolution_from_box_Npx(self, box_size, N_px):
        """
        Compute pixel resolution from box size and number of pixels.

        Parameters
        ----------
        box_size : array_like of float, shape (2,)
        N_px : array_like of int, shape (2,)

        Returns
        -------
        ndarray, shape (2,)
            Resolution per dimension.
        """
        return box_size / N_px

    def box_size_from_N_px_res(self, N_px, resolution):
        """
        Compute box size from pixel counts and resolution.

        Parameters
        ----------
        N_px : array_like of int, shape (2,)
        resolution : array_like of float, shape (2,)

        Returns
        -------
        ndarray, shape (2,)
            Box size per dimension.
        """
        return N_px * resolution

    def N_px_from_box_res(self, box_size, resolution):
        """
        Compute integer pixel counts from box size and resolution.

        Parameters
        ----------
        box_size : array_like of float, shape (2,)
        resolution : array_like of float, shape (2,)

        Returns
        -------
        ndarray of int, shape (2,)
            Number of pixels per dimension.
        """
        return (box_size / resolution).astype(int)

    def _set_boundaries(self):
        """
        Compute symmetric box boundaries around origin.

        Returns
        -------
        ndarray, shape (2,2)
            [[x_min, x_max], [y_min, y_max]]
        """
        bound_vals = self.box_size / 2
        self.boundaries = np.vstack((-bound_vals, bound_vals)).T
        return self.boundaries

    def find_bincentres(self, binedges):
        """
        Compute bin centres from edges.

        Parameters
        ----------
        binedges : ndarray
            Bin edges (1D or 2D for multiple axes).

        Returns
        -------
        ndarray or list of ndarray
            Bin centres for each axis.
        """
        if len(binedges.shape) == 1:
            return 0.5 * (binedges[1:] + binedges[:-1])
        else:
            return [0.5 * (binedges[i][1:] + binedges[i][:-1]) for i in range(binedges.shape[0])]

    def _set_coords(self):
        """
        Compute pixel centres and meshgrid coordinates (X, Y, R).

        Returns
        -------
        tuple of ndarray
            (X, Y) meshgrid arrays of pixel centre coordinates.
        """
        px_bounds = np.array([
            np.linspace(self.boundaries[i][0], self.boundaries[i][1], self.N_px[i] + 1)
            for i in range(2)
        ])
        self.px_bounds = px_bounds
        px_centres = self.find_bincentres(px_bounds)
        X, Y = np.meshgrid(px_centres[0], px_centres[1])
        self.X, self.Y = X, Y
        self.R = self.radial_grid_from_X_Y(X, Y)
        return X, Y

    def radial_grid_from_X_Y(self, X, Y):
        """
        Compute radial distances for a Cartesian grid.

        Parameters
        ----------
        X : ndarray
            X coordinates.
        Y : ndarray
            Y coordinates.

        Returns
        -------
        ndarray
            Radial distances sqrt(X^2 + Y^2).
        """
        return np.sqrt(X**2 + Y**2)

