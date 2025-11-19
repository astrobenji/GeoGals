import pickle
import numpy as np
import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from geogals_io import *
from geogals_log import *

'''
By Tree

Module for storing meta.



'''




class Meta(ABC):
    """
    Abstract base class that provides common metadata handling utilities.

    Subclasses are expected to implement construction from dictionaries and
    to expose a ``parameters`` property describing the run-defining
    parameters.

    Parameters
    ----------
    galaxy_name : str
        Name of the galaxy used to derive file paths.
    results_base_directory : str, optional
        Base directory for results. Default: './results/'.

    Attributes
    ----------
    _fname : str
        Base filename used when saving/loading metadata objects. Defaults to
        'meta'.
    galaxy_name : str
        As provided.
    file_handler : FileHandler
        File handler instance for filesystem interactions.
    logger : Logger
        Logger instance used to record initialization and load/save events.
    """

    def __init__(self, galaxy_name, results_base_directory = './results/'):
        self._fname = 'meta'
        self.galaxy_name = galaxy_name
 

        self.file_handler = FileHandler(galaxy_name, results_base_directory)        
        self.logger = Logger(galaxy_name, results_base_directory)
        self.logger.log_init(self._fname)


    # -------------------------------------------------------------------------
    # Abstract and construction methods
    # -------------------------------------------------------------------------
    @abstractmethod
    def read_meta_from_dictionary(self, metadict):
        """
        Construct the metadata object from a dictionary.

        Subclasses must override this method to parse the specific metadata
        fields they require. The method should populate the instance's
        attributes and return ``self``.

        Parameters
        ----------
        metadict : dict
            Dictionary of metadata fields.
        """
        pass
    
    @property
    @abstractmethod
    def parameters(self):
        """
        Dictionary-like description of the run-defining parameters.

        Subclasses must return a mapping whose keys and values are stable and
        can be used to uniquely identify a run (for example when computing a
        run id hash).
        """
        pass

    def read_meta_from_keywords(self, **kwargs):
        """
        Convenience wrapper that constructs metadata from keyword arguments.

        The semantics are identical to passing the same mapping to
        :meth:`read_meta_from_dictionary`.

        Parameters
        ----------
        **kwargs
            Metadata fields as keyword arguments.

        Returns
        -------
        object
            ``self`` after populating attributes from the supplied keywords.
        """
        return self.read_meta_from_dictionary(kwargs)
    
    def save_to_pickle(self):
        """
        Persist the metadata object to a pickle file in its run directory.
        """
        self.file_handler.pickle_object(self)
    
    @abstractmethod
    def load_from_pickle(self, run_id = None, parameters = None):
        """
        Load and return a metadata object from pickle storage.

        Subclasses must accept either an explicit ``run_id`` or a set of
        identifying ``parameters`` and return a metadata instance.
        """
        pass

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------
    def _compare_attribute_with_argument(self, override=False, **kwargs):
        """
        Ensure that required attributes exist or set them from provided
        keyword values.

        Parameters
        ----------
        override : bool, optional
            If ``True``, existing attributes will be overwritten by provided
            values. If ``False``, only attributes that do not already exist
            will be set. Default: ``False``.
        **kwargs
            Attribute names and values to compare or set.

        Raises
        ------
        AttributeError
            If a required attribute is missing (value is ``None`` and the
            attribute does not already exist on the instance).
        """
        for key, value in kwargs.items():
            if value is None:
                if not hasattr(self, key):
                    raise AttributeError(f"No value for '{key}' stored or provided.")
            else:
                if override or not hasattr(self, key):
                    setattr(self, key, value)




    def _load_header_from_data(self, data_path):
        """
        Extract attributes from a data file header and copy them to the
        instance using :meth:`_compare_attribute_with_argument`.

        Notes
        -----
        This method expects the data file to be compatible with ``h5py`` and
        to expose attributes on the file object. The ``h5py`` import is not
        included in this base module and must be available at runtime if this
        method is used.

        Parameters
        ----------
        data_path : str or pathlib.Path
            Path to the data file from which to extract header attributes.
        """
        
        # data_path = self._fext(f_ext, data_path)

        self._compare_attribute_with_argument(data_path=data_path, override=True)

        header = {}

        with h5py.File(data_path, "r") as f:
            for head in f.attrs.keys():
                header[head] = f.attrs[head]

        self._compare_attribute_with_argument(**header)
    
    def _run_id(self):
        """
        Compute and return a deterministic run identifier from the instance
        ``parameters`` mapping.

        The identifier is the first 8 hex characters of the MD5 hash of the
        JSON-serialized parameters dictionary with stable key ordering. If a
        run directory for the computed id does not exist it will be created
        and an initialization log entry will be emitted.

        Returns
        -------
        str
            The computed run identifier.
        """
        
        import hashlib, json
        s = json.dumps(self.parameters, sort_keys=True).encode()
        run_id = hashlib.md5(s).hexdigest()[:8]
        setattr(self, 'run_id', run_id)

        
        
        if not self.file_handler._get_run_directory(run_id).exists():
            self.file_handler.create_run_directory(run_id)
            self.logger.log_run(run_id)
            self.logger.log_params(run_id, params_dict=self.parameters)

        return run_id
    
    def load_meta(self, meta=None, run_id = None, load_into = None, **search_params):
        """
        Load attributes from another :class:`Meta` instance or from storage.

        Either an existing ``meta`` object must be supplied or a ``run_id`` or
        search ``parameters`` used to locate a saved metadata object. When a
        meta object is copied into ``self``, a log entry is recorded.

        Parameters
        ----------
        meta : Meta, optional
            Existing metadata instance whose attributes will be copied into
            ``self``.
        run_id : str, optional
            Explicit run identifier to load from disk.
        load_into : str, optional
            Optional name of the receiving object used only for logging.
        **search_params
            If ``meta`` is ``None`` and ``run_id`` is not provided, these
            parameters will be used to search for an appropriate run (see
            subclass implementations of :meth:`load_from_pickle`).

        Raises
        ------
        ValueError
            If no metadata object can be loaded from the provided inputs.
        """

        if meta is not None:
            self.__dict__.update(meta.__dict__)
            self.logger.log_load(self.run_id, 'meta', file_type=None, load_into=load_into)

        
        else:
            try:
                meta = self.load_from_pickle(run_id=run_id, parameters=search_params, log=False)
                
                self.load_meta(meta=meta, load_into=load_into)

            except:
                raise ValueError('No meta for this galaxy exists')
        
            # self.logger.log_load(self.run_id, 'meta', file_type=None, load_into=load_into)






# ============================================================================

class SimulationMeta(Meta):
    """
    Metadata container specialised for regularly sampled simulation grids.

    This class stores the grid dimensions in three interchangeable forms:
    ``N_px`` (number of pixels), ``box_size`` (physical size) and
    ``resolution`` (physical size per pixel). At least two of these must be
    supplied when constructing metadata; the third will be computed.

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
    px_bounds : ndarray
        Pixel boundary coordinates used to compute pixel centres.
    X, Y : ndarray
        Meshgrid arrays of pixel centre coordinates.
    R : ndarray
        Radial distances computed from ``X`` and ``Y``.
    """
    # -------------------------------------------------------------------------
    # Object Creation Methods
    # -------------------------------------------------------------------------

    def __init__(self, galaxy_name, results_base_directory='./results/'):
        super().__init__(galaxy_name, results_base_directory)

    

    def read_meta_from_dictionary(self, metadict):
        """
        Populate the instance using fields from a dictionary.

        The input dictionary must supply at least two of the three quantities
        ``N_px``, ``box_size`` and ``resolution``. Missing quantities are
        calculated deterministically from the provided values. After
        populating numeric fields the spatial boundaries and coordinate grids
        are initialised and a run id is computed.

        Parameters
        ----------
        metadict : dict
            Dictionary containing metadata. Expected keys include any
            combination of ``N_px``, ``box_size`` and ``resolution``. Other
            keys are ignored by this base implementation but may be used by
            subclasses.

        Returns
        -------
        SimulationMeta
            ``self`` with computed attributes set.

        Raises
        ------
        KeyError
            If fewer than two of ``N_px``, ``box_size`` and ``resolution``
            are supplied.
        """

        metadict = self._read_meta_params(**metadict)

        # self._set_other_attributes(**metadict)

        # self._compare_attribute_with_argument(**metadict)
        
        
        self._set_boundaries()
        self._set_coords()
        self._run_id()

        return self
    
    @property
    def parameters(self):
        """
        Return a dictionary of parameters that uniquely describe the run.

        The returned mapping contains ``N_px_x``, ``N_px_y``,
        ``resolution_x``, ``resolution_y``, ``box_size_x`` and ``box_size_y``
        with Python-native scalar values (not numpy scalars).
        """
        parameters = {}

        for param in ['N_px', 'resolution', 'box_size']:
            param_value = getattr(self, param)
            

            parameters[param + '_x'] = param_value[0].item()
            parameters[param + '_y'] = param_value[1].item()


        return parameters

    def load_from_pickle(self, run_id=None, parameters=None, log=True):
        """
        Load a SimulationMeta object from pickle storage.

        Parameters
        ----------
        run_id : str, optional
            Explicit run identifier. If provided the corresponding pickle file
            is loaded.
        parameters : dict, optional
            Parameter mapping used to search for a matching run when
            ``run_id`` is not provided. Passed to ``RunIndexer``.
        log : bool, optional
            Passed to :meth:`FileHandler.load_pickle` to control logging.

        Returns
        -------
        SimulationMeta
            Loaded metadata instance.

        Raises
        ------
        ValueError
            If neither ``run_id`` nor ``parameters`` are supplied.
        """
        if run_id is not None:
            return self.file_handler.load_pickle(self._fname, run_id, log)
        elif parameters is not None:
            run_id = RunIndexer(self.galaxy_name, self.file_handler.results_directory.parent).params_to_run(simulation=True, n_matches=2, **parameters)
            return self.load_from_pickle(run_id=run_id, log=log)
            
        else:
            raise ValueError('One of run_id or parameters must not be None.')
        
    
    def _read_meta_params(self, N_px = None, resolution = None, box_size = None, **kwargs):

        """
        Internal helper that validates and sets N_px, resolution and box_size.

        The method requires at least two of the three arguments to be
        non-``None``. It sets corresponding attributes on the instance using
        :meth:`_format_parameters` and computes the missing quantity where
        necessary.

        Parameters
        ----------
        N_px : int or sequence of int, optional
            Number of pixels along each axis.
        resolution : float or sequence of float, optional
            Physical size per pixel.
        box_size : float or sequence of float, optional
            Physical box size along each axis.
        **kwargs
            Additional keyword arguments are accepted but ignored by this
            base implementation.

        Returns
        -------
        dict
            The ``kwargs`` mapping that was passed in (for API compatibility).

        Raises
        ------
        KeyError
            If fewer than two of the primary inputs are provided.
        """

        if np.sum([var is not None for var in [N_px, resolution, box_size]]) < 2:
        # if (np.array([N_px, resolution, box_size]) !=None).sum() <2 :
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
            box_size = self.box_size_from_N_px_res(N_px, resolution)
            self._format_parameters(box_size, 'box_size')

        return kwargs

    



    # -------------------------------------------------------------------------
    # Parameter conversion utilities
    # -------------------------------------------------------------------------
    def resolution_from_box_Npx(self, box_size, N_px):
        """
        Compute pixel resolution from physical box size and pixel counts.

        Parameters
        ----------
        box_size : array_like of float, shape (2,)
            Physical box dimensions.
        N_px : array_like of int, shape (2,)
            Number of pixels along each axis.

        Returns
        -------
        ndarray of float, shape (2,)
            Resolution per dimension (physical size per pixel).
        """
        return box_size / N_px

    def box_size_from_N_px_res(self, N_px, resolution):
        """
        Compute the physical box_size from pixel counts and resolution.

        Parameters
        ----------
        N_px : array_like of int, shape (2,)
            Number of pixels along each axis.
        resolution : array_like of float, shape (2,)
            Physical size per pixel.

        Returns
        -------
        ndarray of float, shape (2,)
            Physical box size per dimension.
        """
        return resolution * N_px

    def N_px_from_box_res(self, box_size, resolution):
        """
        Compute integer pixel counts from box size and resolution.

        Parameters
        ----------
        box_size : array_like of float, shape (2,)
            Physical box dimensions.
        resolution : array_like of float, shape (2,)
            Physical size per pixel.

        Returns
        -------
        ndarray of int, shape (2,)
            Number of pixels per dimension (rounded down via astype(int)).
        """
        return (box_size / resolution).astype(int)
    

    def _format_parameters(self, param, param_name=None):
        """
        Ensure a parameter is represented as a length-2 numpy array and
        optionally assign it to ``self``.

        Scalars will be duplicated to produce a 2-element array. If
        ``param_name`` is ``'N_px'`` the resulting array will be rounded up
        and cast to integer type.

        Parameters
        ----------
        param : int, float or sequence
            Parameter value(s) to format.
        param_name : str, optional
            If provided, the formatted numpy array will be assigned as an
            attribute ``self.<param_name>``. Otherwise the array is returned.

        Returns
        -------
        ndarray
            Formatted parameter array when ``param_name`` is ``None``.
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
        Validate a supplied resolution against the resolution computed from
        ``box_size`` and ``N_px``. If they differ the computed value will
        override the supplied value and a warning is printed.

        Parameters
        ----------
        input_resolution : float or sequence of float
            Supplied resolution values.

        Returns
        -------
        None
            The method updates ``self.resolution`` in-place when necessary.
        """

        input_resolution = self._format_parameters(input_resolution)
        calculated_resolution = self.resolution_from_box_Npx(self.box_size, self.N_px)

        if not np.allclose(input_resolution, calculated_resolution):
             print("Warning: Supplied resolution is inconsistent with box_size and N_px. "
                              "Overriding with calculated value.")
             
        self._format_parameters(calculated_resolution, 'resolution')

        
             


    # -------------------------------------------------------------------------
    # Coordinate and boundary methods
    # -------------------------------------------------------------------------
    def _set_boundaries(self):
        """
        Compute symmetric box boundaries about the origin from ``box_size``.

        Returns
        -------
        ndarray, shape (2,2)
            Array where rows correspond to dimensions and columns to
            [min, max] edges: ``[[x_min, x_max], [y_min, y_max]]``.
        """
        bound_vals = self.box_size / 2
        self.boundaries = np.vstack((-bound_vals, bound_vals)).T
        return self.boundaries

    def find_bincentres(self, binedges):
        """
        Compute bin centres from bin edges.

        Parameters
        ----------
        binedges : ndarray
            Array of bin edges. May be 1-D (single axis) or 2-D (sequence of
            axis edge arrays).

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
        Create pixel boundary arrays, compute pixel centre coordinates and
        populate coordinate grids (X, Y) and radial distances (R).

        Returns
        -------
        tuple
            ``(X, Y)`` meshgrid arrays of pixel centre coordinates.
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
            X coordinates of the grid.
        Y : ndarray
            Y coordinates of the grid.

        Returns
        -------
        ndarray
            Radial distances computed as ``sqrt(X**2 + Y**2)``.
        """
        return np.sqrt(X**2 + Y**2)




