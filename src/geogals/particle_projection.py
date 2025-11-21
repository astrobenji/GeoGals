from geogals_io import *
from geogals_log import *
import h5py
from meta import *
import utilities as ut
import numpy as np
import matplotlib.pyplot as plt
from metallicity_maps import MetMap
import copy


class ParticleData(SimulationMeta):
    """
    Class for creating and manipulating 2D projections from simulated particle data.

    Extends `SimulationMeta` to include functionality for reading,
    transforming, and projecting 3D particle data into 2D maps such as surface
    density or metallicity projections.

    Notes
    -----
    The class expects particle arrays to follow a shape convention of
    ``(N_particles, 3)`` for position-like arrays (x, y, z) and ``(N_particles,)``
    for scalar particle properties (e.g. mass, metallicity). Projection and
    binning use the spatial metadata provided by :class:`SimulationMeta`.
    """

    def __init__(self, galaxy_name, results_base_directory='./results', meta=None, run_id = None, **parameters):
        """
        Initialize a ParticleData object.

        Parameters
        ----------
        galaxy_name : str
            Name of the galaxy or simulation to associate with this dataset.
        results_base_directory : str, optional
            Base results directory. Default './results'.
        meta : SimulationMeta, optional
            Simulation metadata object to use for the projection.
        run_id : str, optional
            Explicit run identifier to load metadata for.
        **parameters
            Additional keyword arguments forwarded to :meth:`load_meta`.

        Notes
        -----
        This constructor calls :class:`SimulationMeta.__init__` then loads
        metadata via :meth:`load_meta`. It also sets ``_fname`` to
        ``'particle_data'`` so pickling and file routines use that base name.
        """
        super().__init__(galaxy_name, results_base_directory)
        
        self.load_meta(meta, run_id, load_into='particle_data', **parameters)
        self._fname = 'particle_data'


    def from_datadict(self, datadict, centre=[0, 0, 0], z_bound=np.inf, rotation_angles=None, position_key='position', **kwargs):
        """
        Create the object from a dictionary of particle data.

        All data are stored as attributes using the keys provided in the dictionary.
        The method filters particles to those that fall within the 2D spatial
        boundaries (from :attr:`boundaries`) and within the z-bound given by
        ``z_bound``. It optionally rotates particle coordinates before applying
        the spatial selection.

        Parameters
        ----------
        datadict : dict
            Dictionary containing particle data arrays. Must include the key
            specified by ``position_key`` which should contain an array of
            shape ``(N_particles, 3)`` giving (x, y, z) coordinates.
        centre : array_like, optional
            Coordinates of the galaxy centre. Subtracted from particle positions
            before any rotation or selection. Default is ``[0, 0, 0]``.
        z_bound : float, optional
            Maximum absolute z-distance for particles to be included in the
            projection. Particles with ``-z_bound < z < z_bound`` are kept.
            Default is ``np.inf`` (no z clipping).
        rotation_angles : array_like of float, optional
            Sequence of rotation angles (radians) about the x, y and z axes to
            apply to particle coordinates before selection. If ``None``, no
            rotation by angles is performed.
        position_key : str, optional
            Key in ``datadict`` that holds particle positions. Default
            ``'position'``.
        **kwargs
            Additional options:
            - data_function : callable, optional
                Callable applied to ``datadict`` for custom preprocessing.
                If present and the instance has ``data_header``, it is called
                as ``data_function(datadict, data_header)`` else
                ``data_function(datadict)``.
            - rotation_tensor : array_like, optional
                Alternative to ``rotation_angles``: a rotation matrix/tensor
                passed to :meth:`_rotate`.

        Side effects
        ------------
        - Sets attributes:
          - ``initial_n_particles`` : int
            Number of particles in the input (before masking).
          - ``included_particles`` : boolean ndarray
            Mask of particles retained after spatial and z clipping.
          - ``n_particles_in_bounds`` : int
            Number of particles within the spatial + z bounds.
          - ``position`` : ndarray
            Positions of included particles (shape ``(n_included, 3)``).
          - other properties from ``datadict`` restricted to included particles.

        Raises
        ------
        AssertionError
            If ``position_key`` is not present in ``datadict``.
        """
        
        datadict = copy.deepcopy(datadict)
        if 'data_function' in kwargs.keys():
            if hasattr(self, 'data_header'):
                datadict = kwargs['data_function'](datadict, self.data_header)
            else:
                datadict = kwargs['data_function'](datadict)

        assert position_key in datadict.keys()

        centre = np.array(centre)
        pos = datadict[position_key] - centre

        if rotation_angles is not None:
            pos = self._rotate(pos, rotation_angles)
        if 'rotation_tensor' in kwargs.keys():
            pos = self._rotate(pos, rotation_tensor=kwargs['rotation_tensor'])

        included_particles = np.prod(
            [(pos[:, i] > self.boundaries[i][0]) * (pos[:, i] < self.boundaries[i][1]) for i in range(2)],
            axis=0
        )
        included_particles *= (pos[:, 2] > -z_bound) * (pos[:, 2] < z_bound)
        included_particles = included_particles.astype(bool)
        setattr(self, 'initial_n_particles', len(included_particles))
        setattr(self, 'included_particles', included_particles)
        setattr(self, 'n_particles_in_bounds', included_particles.sum())

        pos = pos[included_particles]
        self._compare_attribute_with_argument(position=pos, override=True)

        datadict.pop(position_key)
        datadict = {key: val[included_particles] for key, val in datadict.items()}

        self._compare_attribute_with_argument(**datadict, override=True)
        self._bin_positions()

    def load_particle_data_from_file(self, data_path, centre=[0, 0, 0], z_bound=np.inf, rotation_angles=None, position_key='position', **kwargs):
        """
        Load particle data from file.

        Header attributes are loaded into metadata and particle data into attributes.

        Parameters
        ----------
        data_path : str or pathlib.Path
            Path to the file containing particle data. Supported formats are
            handled by the instance's :class:`FileHandler`.
        centre : array_like, optional
            Centre coordinates to apply to loaded positions before selection.
            See :meth:`from_datadict`.
        z_bound : float, optional
            z clipping threshold for particle inclusion.
        rotation_angles : array_like, optional
            Rotation angles passed to :meth:`from_datadict`.
        position_key : str, optional
            Key name for positions within the loaded data dictionary.
        **kwargs
            Additional keyword arguments forwarded to :meth:`file_handler.load_data`
            and to :meth:`from_datadict` (for example ``data_function`` or
            ``rotation_tensor``).

        Returns
        -------
        None

        Notes
        -----
        If the loaded ``data`` is a dictionary, this function calls
        :meth:`from_datadict` to populate the object. Any header returned by the
        loader is currently not directly stored here (but could be used to
        populate metadata via :meth:`_store_header_from_dict` if desired).
        """
        data_path = Path(data_path)

        data, header = self.file_handler.load_data(data_path, **kwargs)
        if isinstance(data, dict):
            self.from_datadict(data, centre=centre, z_bound=z_bound, rotation_angles=rotation_angles, position_key = position_key, **kwargs)

        # return header
        # do something with header



    def _bin_positions(self, pos = None):
        """
        Compute and store the pixel bin index of each particle.

        The bin indices are computed along the x and y dimensions relative to
        the instance's pixel boundaries (``px_bounds``) and number of pixels
        (``N_px``). If ``pos`` is provided, the function returns the computed
        bin indices for the supplied positions instead of storing them.

        Parameters
        ----------
        pos : array_like, optional
            Positions to bin of shape ``(N, 3)``. If ``None``, the instance's
            ``position`` attribute (set by :meth:`from_datadict`) is used.

        Returns
        -------
        None or tuple(ndarray, ndarray)
            If ``pos`` is ``None`` the method stores ``x_bindices`` and
            ``y_bindices`` attributes and returns ``None``. If ``pos`` is
            provided, it returns ``(x_bindices, y_bindices)`` arrays.

        Notes
        -----
        Indices are zero-based and clipped to the pixel grid via
        ``np.digitize`` behaviour. Values equal to the rightmost edge will
        produce index ``N_px-1`` when used in downstream indexing.
        """
        if pos is None:
            x, y = (self.position[:, :-1]).T
        else:
            x,y = (pos[:, :-1]).T

        x_bindices = np.digitize(x, self.px_bounds[0]) - 1
        y_bindices = np.digitize(y, self.px_bounds[1]) - 1

        if pos is None:
            setattr(self, 'x_bindices', x_bindices)
            setattr(self, 'y_bindices', y_bindices)
        else:
            return x_bindices, y_bindices

    def _rotate(self, pos, rotation_angles = None, rotation_tensor = None):
        """
        Rotate coordinates using predefined utility functions.

        Parameters
        ----------
        pos : array_like
            Particle positions of shape (N_particles, 3).
        rotation_angles : array_like of float, optional
            Angles (radians) to rotate about x, y, and z axes. If provided,
            rotation is performed using :mod:`utilities.coordinate.get_coordinates_rotated`.
        rotation_tensor : array_like, optional
            A rotation matrix/tensor. If provided, it is passed to the same
            utility as ``rotation_angles``.

        Returns
        -------
        ndarray
            Rotated particle positions of shape ``(N_particles, 3)``.

        Raises
        ------
        ValueError
            If neither ``rotation_angles`` nor ``rotation_tensor`` is provided.
        """
        if rotation_angles is not None:
            pos = ut.coordinate.get_coordinates_rotated(pos, rotation_angles=rotation_angles)
        elif rotation_tensor is not None:
            pos = ut.coordinate.get_coordinates_rotated(pos, rotation_tensor=rotation_tensor)
        else:
            raise ValueError

        return pos

    def _project_values(self, values, pos = None,per_unit_area=False):
        """
        Project particle values into a 2D pixel grid.

        Binning is performed by computing integer pixel indices for particles
        and summing values into pixels using ``np.bincount``. If ``per_unit_area``
        is True the returned map is divided by the pixel area computed from
        ``self.resolution``.

        Parameters
        ----------
        values : array_like
            Array of particle values to project. Must have length equal to the
            number of particles currently represented in the object (or the
            length of ``pos`` if supplied).
        pos : array_like, optional
            Alternative positions corresponding to ``values`` (shape ``(N,3)``).
            If provided, positions are binned from these coordinates; otherwise
            stored indices (``x_bindices``, ``y_bindices``) are used.
        per_unit_area : bool, optional
            If True, divide projected values by the pixel area (``resolution_x * resolution_y``).
            Default is False.

        Returns
        -------
        ndarray
            2D array (shape ``N_px``) of projected values.
        """
        px_area = 1
        if per_unit_area:
            px_area = np.prod(self.resolution)
        if pos is not None:
            x_bindices, y_bindices = self._bin_positions(pos=pos)
            multi_idx = np.ravel_multi_index((x_bindices, y_bindices), self.N_px, mode='clip')
        else:
            multi_idx = np.ravel_multi_index((self.x_bindices, self.y_bindices), self.N_px)

        return np.bincount(multi_idx, weights=values, minlength=self.N_px.prod()).reshape(self.N_px) / px_area

    def _property(self, property):
        """
        Retrieve a stored particle property by name.

        Parameters
        ----------
        property : str
            Name of the property to retrieve.

        Returns
        -------
        ndarray
            Array of the requested property values.

        Raises
        ------
        AttributeError
            If the property is not found in the object.
        """
        if hasattr(self, property):
            vals = getattr(self, property)
        else:
            raise AttributeError(f'no property named {property}')
        return vals

    def _project(self, property_name, pos = None, per_unit_area=False):
        """
        Project a particle property into 2D.

        Convenience wrapper that fetches the property array by name and calls
        :meth:`_project_values`.

        Parameters
        ----------
        property_name : str
            Name of the particle property to project.
        pos : array_like, optional
            Alternative positions to use for binning (see :meth:`_project_values`).
        per_unit_area : bool, optional
            If True, normalize projected values by pixel area. Default is False.

        Returns
        -------
        ndarray
            2D projected map of the specified property.
        """
        vals = self._property(property_name)
        return self._project_values(vals, pos, per_unit_area=per_unit_area)

    def _get_weights(self, **kwargs):
        """
        Retrieve particle weights from specified functions or properties.

        The method supports two mutually-compatible mechanisms for obtaining
        weights:
        - ``weight_property``: a string or list of strings naming attributes on
          the object to use as weight arrays.
        - ``weights_function``: a callable that accepts the object's ``__dict__``
          and returns an array-like weight array.

        Parameters
        ----------
        **kwargs
            weight_property : str or list of str, optional
                Property name(s) to use as weights.
            weights_function : callable, optional
                Function returning weights given the object's data.

        Returns
        -------
        list of ndarray
            A list of weight arrays. Each entry corresponds to one weight term.

        Raises
        ------
        AssertionError
            If neither ``weight_property`` nor ``weights_function`` is provided.
        """
        weights = []
        if 'weight_property' in kwargs.keys():
            if type(kwargs['weight_property']) == str:
                weights.append(self._property(kwargs['weight_property']))
            if type(kwargs['weight_property']) == list:
                weights = [self._property(weight_property) for weight_property in kwargs['weight_property']]

        if 'weights_function' in kwargs.keys():
            weights.append(kwargs['weights_function'](self.__dict__))

        if 'weights_function' not in kwargs.keys() and 'weight_property' not in kwargs.keys():
            raise AssertionError('must include weights_function or weight_property as keyword argument')

        return weights

    def _weighted_project(self, property_name, pos = None, per_unit_area=False, **kwargs):
        """
        Perform a weighted projection of a particle property.

        The projection is weighted by one or more particle properties or
        by the result of a weight function. When multiple weight terms are
        provided, the numerator is computed as the projection of
        ``property * prod_i(weights_i)`` and the denominator as the
        product over i of the projection of each ``weights_i``; the final map
        is their ratio.

        Parameters
        ----------
        property_name : str
            Name of the particle property to project.
        pos : array_like, optional
            Positions to use for binning.
        per_unit_area : bool, optional
            If True, normalize projected values by pixel area.
        **kwargs
            weight_property : str or list of str, optional
                Property name(s) used as weights.
            weights_function : callable, optional
                Callable returning weight arrays.

        Returns
        -------
        ndarray
            Weighted 2D projection of the given property.

        Notes
        -----
        The implementation computes:
            numerator = projection(property * prod(weights_i))
            denominator = prod_i(projection(weights_i))
        and returns numerator / denominator. This matches a multiplicative
        weighting scheme; ensure this is appropriate for your use-case.
        """
        vals = self._property(property_name)
        weights = self._get_weights(**kwargs)

        return self._project_values(vals * np.prod(weights, axis=0), pos, per_unit_area=per_unit_area) / np.prod(
            [self._project_values(weight, pos, per_unit_area=per_unit_area) for weight in weights], axis=0)

    def project(self, property_name, per_unit_area=False, **kwargs):
        """
        Compute the (optionally weighted) projection of a particle property.

        This method handles special logic for metallicity (ensuring mass is
        included as a weighting term unless explicitly provided otherwise).
        It supports:
        - weight_property: specify properties to weight by
        - weights_function: provide a callable that returns weights
        - solar_units: when projecting metallicity, convert to solar units
        - dex: convert result to log10 scale

        Parameters
        ----------
        property_name : str
            Name of the particle property to project.
        per_unit_area : bool, optional
            If True, normalize by pixel area. Default is False.
        **kwargs
            weight_property : str or list, optional
                Property or properties to use as weights.
            weights_function : callable, optional
                Function returning weights (array-like).
            solar_units : bool, optional
                If True and projecting metallicity, convert to solar units.
            dex : bool, optional
                If True, return the base-10 logarithm of the projection.

        Returns
        -------
        ndarray
            2D projection of the requested property.
        """
        if 'rotation_angles' in kwargs.keys():
            pos = self._rotate(self.position, rotation_angles=kwargs['rotation_angles'])
        else:
            pos = None

        Z = False

        if property_name.lower() == 'metallicity':
            Z = True
            if 'weight_property' not in kwargs.keys():
                kwargs['weight_property'] = 'mass'
            else:
                weight_props = kwargs['weight_property']
                if type(weight_props) == str and weight_props != 'mass':
                    weight_props = [weight_props] + ['mass']
                elif 'mass' not in weight_props:
                    weight_props += ['mass']
                kwargs['weight_property'] = weight_props

        if 'weights_function' in kwargs.keys() or 'weight_property' in kwargs.keys():
            projection = self._weighted_project(property_name, pos, per_unit_area, **kwargs)
        else:
            projection = self._project(property_name, pos, per_unit_area)

        if Z:
            if 'solar_units' in kwargs.keys():
                if kwargs['solar_units']:
                    projection /= 0.0139  # metallicity relative to solar

        # log units
        if 'dex' in kwargs.keys():
            if kwargs['dex']:
                projection = np.log10(projection)

        return projection

    def assign_property(self, property_name, **kwargs):
        """
        Add a property from either an array of values or a function that defines a composite property.

        Parameters
        ----------
        property_name : str
            The key to use for the property.
        **kwargs
            property_function : callable, optional
                A function that takes a dictionary as an argument and composes
                attributes of this object to create a new property. The function
                should return an array with length equal to the number of
                particles included in the object.
            property_values : array_like, optional
                An array of values to assign as an attribute. The length must
                either match the original input particle count or the number of
                particles within bounds; see :meth:`_add_prop_vals`.

        Notes
        -----
        If both ``property_function`` and ``property_values`` are supplied,
        both will be applied (function first, then values).
        """
        if 'property_function' in kwargs.keys():
            self._add_prop_func(property_name, **kwargs)

        if 'property_values' in kwargs.keys():
            self._add_prop_vals(property_name, **kwargs)

    def _add_prop_func(self, property_name, property_function):
        """
        Add a computed property using a function.

        Parameters
        ----------
        property_name : str
            Name of the property to add.
        property_function : callable
            Function that takes the object's ``__dict__`` and returns an array
            of values for the included particles.

        Returns
        -------
        None

        Side effects
        ------------
        - Sets ``self.<property_name>`` to the returned array.
        """
        setattr(self, property_name, property_function(self.__dict__))

    def _add_prop_vals(self, property_name, property_values):
        """
        Add a property from a provided array of values.

        The length of ``property_values`` must match either the original particle
        count (before bounding) or the number of particles within the bounds.

        Parameters
        ----------
        property_name : str
            Name of the property to add.
        property_values : array_like
            Values for the property.

        Raises
        ------
        ValueError
            If ``property_values`` length does not match expected lengths.
        """
        val_len = len(property_values)
        if val_len == self.initial_n_particles:
            setattr(self, property_name, property_values[self.included_particles])
        elif val_len == self.n_particles_in_bounds:
            setattr(self, property_name, property_values)
        else:
            raise ValueError(
                f'property_values argument has length {val_len} but should have the same length as original particle data ({self.initial_n_particles}) or the length of particle data within the bounds ({self.n_particles_in_bounds})'
            )

    def multiple_projections(self, projection_dict):
        """
        Compute multiple 2D projections in a single call.

        Parameters
        ----------
        projection_dict : dict
            Dictionary defining multiple projections.
            Each key is a projection name, and each value is a dictionary of
            arguments to pass to :meth:`project`.

        Returns
        -------
        dict
            Dictionary mapping projection names to their 2D projection arrays.

        Example
        -------
        projection_dict = {
            'density': {'property_name': 'mass', 'per_unit_area': True},
            'metallicity': {'property_name': 'metallicity', 'dex': True}
        }
        """
        projection_results = {}
        for projection_name, projection_info in projection_dict.items():
            projection_results[projection_name] = self.project(**projection_info)

        return projection_results


    def create_metmap_obj(self, projection_dict):
        """
        Create and return a MetMap object from multiple projections.

        Parameters
        ----------
        projection_dict : dict
            Dictionary specifying multiple projections to compute. Passed to
            :meth:`multiple_projections`.

        Returns
        -------
        MetMap
            A MetMap instance initialized from the computed projections.

        Notes
        -----
        If a MetMap cannot be created using the current ``run_id``, the method
        will attempt to construct a temporary SimulationMeta from the current
        ``box_size`` and ``resolution`` and instantiate the MetMap with that metadata.
        """
        projection_results = self.multiple_projections(projection_dict)
        try:
            metmap = MetMap(galaxy_name=self.galaxy_name, run_id = self.run_id).from_dictionary(projection_results)
        except:
            meta = SimulationMeta(self.galaxy_name).read_meta_from_keywords(box_size=self.box_size, resolution=self.resolution)
            metmap = MetMap(galaxy_name=self.galaxy_name, meta = meta).from_dictionary(projection_results)
        return metmap

    def plot_projection(self, property_name, per_unit_area=False, save=True, **kwargs):
        """
        Plot a 2D projection of a particle property.

        Parameters
        ----------
        property_name : str
            Name of the particle property to project and plot.
        per_unit_area : bool, optional
            If True, normalize by pixel area. Default is False.
        save : bool, optional
            If True, save the produced figure using the file handler. Default True.
        **kwargs
            Additional keyword arguments passed to `imshow` and labeling
            helper functions. Common keys:
            - 'imshow_*' keys are forwarded to ``ax.imshow`` after stripping the prefix.
            - 'fig_*' keys are forwarded to ``plt.figure``.
            - labeling keys (distance_units, property_units, dex, solar_units, etc.)

        Notes
        -----
        This function uses Matplotlib to display the projection and then calls
        :meth:`_label_projection_plot` to set axis labels and colorbar. When
        ``save`` is True the image filename is created with
        :meth:`_get_projection_filename` and saved via the :class:`FileHandler`.
        """
        projection = self.project(property_name, per_unit_area, **kwargs)

        fig_dict = {key: val for key, val in kwargs.items() if 'fig' in key}
        fig = plt.figure(**fig_dict)
        ax = fig.add_subplot()

        imshow_dict = {key.replace('imshow_', ''): val for key, val in kwargs.items() if 'imshow' in key}
        im = ax.imshow(projection, extent=self.boundaries.flatten(), **imshow_dict)

        self._label_projection_plot(im, ax, property_name, per_unit_area, **kwargs)

        if save:
            f_name = self._get_projection_filename(property_name, **kwargs)
            self.file_handler.save_plot(f_name, self.run_id)
            self.logger.log_save(self.run_id, f_name, 'png')

    def _projection_labels(self, property_name, per_unit_area, **kwargs):
        """
        Generate axis and colorbar labels for a projection plot.

        This helper composes sensible default axis labels and a colorbar label
        based on the `property_name` and provided keyword arguments. It reads
        optional instance attributes (e.g. ``distance_units``) when available
        to form units-aware labels.

        Parameters
        ----------
        property_name : str
            Name of the projected property.
        per_unit_area : bool
            Whether the projection was computed per unit area.
        **kwargs
            Optional keyword arguments that can override default labels:
            - distance_units : str
                Units for x and y axes.
            - property_units : str
                Units for the projected property.
            - solar_units : bool
                If True, express metallicity relative to solar.
            - dex : bool
                If True, apply logarithmic (dex) labeling.
            - x_label, y_label, cbar_label : str
                Custom labels.

        Returns
        -------
        dict
            Dictionary containing formatted axis and colorbar labels with keys:
            'x_label', 'y_label', 'cbar_label'.
        """
        label_dict = {}
        if 'distance_units' in kwargs.keys():
            distance_units = kwargs['distance_units']
        elif hasattr(self, 'distance_units'):
            distance_units = self.distance_units
        else:
            distance_units = 'arbitrary units'

        for axis in ['x', 'y']:
            if axis + '_label' in kwargs.keys():
                label_dict[axis + '_label'] = kwargs[axis + '_label']
            else:
                label_dict[axis + '_label'] = f'{axis} ({distance_units})'

        if 'cbar_label' in kwargs.keys():
            label_dict['cbar_label'] = kwargs['cbar_label']

        else:
            if property_name == 'metallicity':
                if 'solar_units' in kwargs.keys():
                    if kwargs['solar_units']:
                        cbar_label = r'$Z/Z_\odot$'
                    else:
                        cbar_label = r'$Z$'
                else:
                    cbar_label = r'$Z$'
                if 'dex' in kwargs.keys():
                    if kwargs['dex']:
                        cbar_label = f"$\\log\\left({cbar_label.replace('$', '')}\\right)$"

            elif property_name == 'mass':
                cbar_label = r'$M$'
                if 'property_units' in kwargs.keys():
                    if kwargs['property_units'] == 'solar mass':
                        cbar_label = r'$M/M_\odot$'
                        cbar_units = None
                    else:
                        cbar_units = kwargs['property_units']
                else:
                    cbar_units = 'mass units'

                if per_unit_area:
                    cbar_units += f'({distance_units})' + r'$^{-2}$'
                if cbar_units is not None:
                    cbar_label = f'{cbar_label} ({cbar_units})'

            else:
                cbar_label = property_name
                if 'property_units' in kwargs.keys():
                    cbar_label += kwargs['property_units']
                else:
                    cbar_units += 'arbitrary units'

                if per_unit_area:
                    cbar_units += f'({distance_units})' + r'$^2$'

                cbar_label = f'{cbar_label} ({cbar_units})'

            label_dict['cbar_label'] = cbar_label

            return label_dict

    def _label_projection_plot(self, im, ax, property_name, per_unit_area, **kwargs):
        """
        Apply axis and colorbar labels to a projection plot.

        Parameters
        ----------
        im : AxesImage
            The image object returned by ``ax.imshow``.
        ax : matplotlib.axes.Axes
            The axes to which labels and colorbar should be attached.
        property_name : str
            Name of the projected property.
        per_unit_area : bool
            Whether the projection was computed per unit area.
        **kwargs
            Keyword arguments passed to :meth:`_projection_labels` and to the
            colorbar creation routine (prefixed with ``cbar_`` in the dict).

        Notes
        -----
        This function updates the provided Axes with axis labels and creates
        a colorbar whose title is taken from the label dictionary returned by
        :meth:`_projection_labels`.
        """
        label_dict = self._projection_labels(property_name, per_unit_area, **kwargs)

        cbar_dict = {key.replace('cbar_', ''): val for key, val in label_dict.items() if 'cbar' in key}
        cbar = plt.colorbar(im, ax=ax, **cbar_dict)

        ax.set_xlabel(label_dict['x_label'])
        ax.set_ylabel(label_dict['y_label'])


    def _get_projection_filename(self, property_name, **kwargs):
        """
        Generate a filename for saving a projected property.

        The filename incorporates the property name and, optionally, the weight
        property used for weighted projections to make filenames informative.

        Parameters
        ----------
        property_name : str
            Name of the property being projected.
        **kwargs
            Optional keyword arguments that control filename behavior:
            - filename : str, optional
                If provided, use this string as the filename.
            - weight_property : str or list of str, optional
                Property (or list of properties) used as a weighting parameter.
                Mass is treated specially and is omitted from the appended
                label for metallicity projections.

        Returns
        -------
        str
            The generated filename, including weighting information if applicable.
        """
        if 'filename' in kwargs.keys():
            fname = kwargs['filename']
        else:
            fname = property_name
            if 'weight_property' in kwargs.keys():
                if type(kwargs['weight_property']) == str:
                    if property_name == 'metallicity' and kwargs['weight_property'] == 'mass':
                        pass
                    else:
                        fname += f'_weighted_by_{kwargs['weight_property']}'
                elif type(kwargs['weight_property']) == list:
                    if property_name == 'metallicity' and 'mass' in kwargs['weight_property']:
                        kwargs['weight_property'].remove('mass')
                    fname += f'_weighted_by_{'_'.join(kwargs['weight_property'])}'
        return fname
