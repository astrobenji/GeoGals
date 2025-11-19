import pickle
import numpy as np
import datetime
from pathlib import Path
from geogals_log import *
from astropy.io import fits
import h5py

''' 
By Tree

Module to assist with file handling and directory paths. The FileHandler class
manages the results directory structure as well as loading data and saving
results to file.

Results directory is organised as

results/
│
├── galaxy/                 
│   │
│   ├── run_XXXXXXXX/                # Run specific directory with run id*
│   │   |
│   │   └── meta.pkl                 # Saved metadata for the run
│   │   └── ...                      # Any other data output for the run
│   │
│   └── log.log                      # Log file generated for this galaxy

To be updated as more directory structure added.

* Runs are given a unique identifier which is hashed from the parameters which
define the meta class for that run

There is also a RunIndexer class to provide convenience in searching through
runs and converting between run ids and parameters.

'''

class RunIndexer:
    """
    Scan existing run directories and provide utilities to match runs by
    parameter values.

    All methods by Tree.

    Parameters
    ----------
    galaxy_name : str
        Galaxy name used by the underlying :class:`FileHandler`.
    results_base_directory : str, optional
        Base results directory. Default: './results'.

    Attributes
    ----------
    file_handler : FileHandler
        File handler instance for interacting with the filesystem.
    runs : dict
        Mapping of ``run_id`` -> parameter dictionary loaded from the run's
        metadata.
    """

    def __init__(self, galaxy_name, results_base_directory='./results'):
        self.file_handler = FileHandler(galaxy_name, results_base_directory)

        self.runs = self._run_scan()
    
    def _run_scan(self):
        """
        Inspect the results directory and build a mapping of run identifiers to
        their parameter dictionaries.

        Returns
        -------
        dict
            Mapping ``run_id`` (str) -> parameters (dict).
        """
        
        runs = {}

        for run_dir in self.file_handler.results_directory.iterdir():
            if not run_dir.is_dir():
                continue
            if 'run' not in run_dir.name:
                continue

            run_id = run_dir.name.split('_')[1]
            
            meta = self.file_handler.load_pickle('meta', run_id)

            runs[run_id] = meta.parameters

        return runs
    
    def _convert_sim_params(self, **params):
        """
        Convert scalar or short-form simulation parameters to explicit x/y
        parameter pairs.

        The input may contain entries such as ``box_size=10`` or
        ``N_px=[256,512]``. Scalars are duplicated to form length-2 arrays.

        Parameters
        ----------
        **params
            Arbitrary parameter keyword arguments.

        Returns
        -------
        dict
            New parameter dictionary where each parameter ``p`` is replaced by
            ``p_x`` and ``p_y`` with numeric (Python-native) values.

        Raises
        ------
        AssertionError
            If any provided parameter expands to more than two elements.
        """
        new_parameters = {}
        for parameter_key, parameter_value in params.items():
            param = np.array([parameter_value]).flatten()
            if len(param) == 1:
                param = np.array([param[0], param[0]])
            assert len(param) == 2, "Too many dimensions supplied."
            new_parameters[parameter_key + '_x'] = param[0].item()
            new_parameters[parameter_key + '_y'] = param[1].item()

        return new_parameters
        

    
    def params_to_run(self, simulation = False, n_matches = 2, **search_params):
        """
        Return run identifiers whose stored parameters match the provided
        search parameters.

        Parameters
        ----------
        simulation : bool, optional
            If ``True``, the provided ``search_params`` will be converted to
            x/y pairs (see :meth:`_convert_sim_params`) and the required
            number of matching fields will be doubled. Default is ``False``.
        n_matches : int, optional
            Minimum number of matching parameter key/value pairs required for
            a run to be considered a match. Default is 2.
        **search_params
            Parameter key/value pairs to match against stored runs.

        Returns
        -------
        str or list
            If exactly one run matches, returns its ``run_id`` as a string.
            Otherwise returns a list of matching ``run_id`` strings (possibly
            empty).
        """
        runs = []
        if simulation:
            search_params = self._convert_sim_params(**search_params)
            n_matches *= 2

        
        for run_id, run_params in self.runs.items():
            match_params = set(search_params.items()) & set(run_params.items())

        
            if len(match_params) >= n_matches:
                runs.append(run_id)

        if len(runs) == 1:
            return runs[0]
        
        else:
            return runs
        



class FileHandler:
    """
    Filesystem utilities for creating run directories and reading/writing
    pickled objects.
    
    All methods by Tree.

    Parameters
    ----------
    galaxy_name : str
        Galaxy name used to derive the results directory.
    results_base_directory : str, optional
        Root directory for results. Default is './results'.
    

    Attributes
    ----------
    galaxy_name : str
        As provided.
    results_directory : pathlib.Path
        Path to the galaxy-specific results directory.
    """

    def __init__(self, galaxy_name, results_base_directory='./results'):
        self.galaxy_name = galaxy_name
        self.results_directory = self._create_results_directory(results_base_directory)
    
    def _create_results_directory(self, results_base_directory='./results'):

        """
        Ensure the galaxy results directory exists, creating it if necessary.

        Parameters
        ----------
        results_base_directory : str, optional
            Base directory in which the galaxy directory will be placed. This
            method returns the full path to the created or existing directory.

        Returns
        -------
        pathlib.Path
            The created or existing results directory path.
        """

        results_directory = Path(results_base_directory) / self.galaxy_name


        if not results_directory.exists():
            results_directory.mkdir(parents=True)

        return results_directory


    def create_run_directory(self,run_id):

        """
        Create a directory for a specific run within the galaxy results
        directory.

        Parameters
        ----------
        run_id : str
            Identifier for the run. The directory name will be ``run_{run_id}``.

        Returns
        -------
        pathlib.Path
            Path to the run directory.
        """

        run_directory =  self._get_run_directory(run_id)

        if not run_directory.exists():
            run_directory.mkdir()
            
        

        return run_directory
    
    def _get_run_directory(self,run_id):
        """
        Compute the path to a run directory without creating it.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        pathlib.Path
            Path to ``results_directory / f'run_{run_id}'``.
        """
        return self.results_directory / f'run_{run_id}'

    def load_pickle(self, fname, run_id, log=True):

        """
        Load a pickled object from the specified run directory.

        Parameters
        ----------
        fname : str
            Base filename (without extension) of the pickled object.
        run_id : str
            Run identifier where the file is stored.
        log : bool, optional
            If ``True``, call the loaded object's logger to record the
            load operation. Default: ``True``.

        Returns
        -------
        object
            The unpickled Python object from the file.
        """

        path = self.results_directory / f'run_{run_id}' / f'{fname}.pkl'


        with open(path, "rb") as f:
            obj = pickle.load(f)
        if log:
            obj.logger.log_read(obj.run_id, obj._fname, 'pickle')

        return obj
    
    def pickle_object(self, obj):
        """
        Pickle an object into its run directory using the object's
        ``run_id`` and ``_fname`` attributes.

        Parameters
        ----------
        obj : object
            Object to pickle. Must expose ``run_id`` and ``_fname`` attributes
            and a ``logger`` that implements :meth:`Logger.log_save`.

        Returns
        -------
        object
            The same object that was passed in (returned for convenience).
        """
        path = self.results_directory / f'run_{obj.run_id}' / f'{obj._fname}.pkl'
        with open(path, "wb") as f:
            pickle.dump(obj, f)


        obj.logger.log_save(obj.run_id, obj._fname, 'pickle')
        return obj
    
    def open_data(self, data_path, **kwargs):
        """
        Open a data file and return its contents using the appropriate loader
        based on file suffix.

        This method dispatches to one of the internal file-specific loaders
        (FITS, HDF5, NumPy, pickle, etc.) depending on the file extension.  
        Additional keyword arguments are passed directly to the selected loader.

        Parameters
        ----------
        data_path : str or pathlib.Path
            Path to the data file to open.

        **kwargs
            Additional keyword arguments forwarded to the specific open function.
            These depend on the file type. For example, ``numpy.load`` may accept
            ``allow_pickle=True``.

        Returns
        -------
        object
            The parsed data structure produced by the corresponding loader.
            The exact return type depends on the file format:

            - FITS → ``(list of numpy.ndarray, list of fits.Header)``
            - HDF5 → ``dict`` (optionally with a ``header`` dict)
            - NPY  → ``numpy.ndarray``
            - PKL  → user-defined Python object

        Notes
        -----
        Unsupported file extensions will return ``None``.
        """
        data_path = Path(data_path)

        open_method = self._find_open_method_from_suffix(data_path.suffix.lower())
        return open_method(data_path, **kwargs)
                

    def _find_open_method_from_suffix(self, ext):
        """
        Return an appropriate file-opening function for the given file extension.

        Parameters
        ----------
        ext : str
            File extension including the leading dot (e.g., ``".fits"``).

        Returns
        -------
        callable
            A method or function capable of opening the file type.

        Raises
        ------
        KeyError
            If the extension does not correspond to a known data format.
        """
        if ext in [".fits", ".fit", ".fts"]:
            return self._open_fits
        elif ext in [".h5", ".hdf5"]:
            return self._open_hdf5
        elif ext == ".npy":
            return np.load
        elif ext == ".pkl":
            return self._open_pickle_data
        else:
            raise KeyError(f"Unsupported file extension: {ext!r}")


    def _open_fits(self, data_path):
        """
        Open a FITS file and return its data and headers.

        Parameters
        ----------
        data_path : str or pathlib.Path
            Path to the FITS file.

        Returns
        -------
        tuple
            A two-element tuple ``(data, header)`` where:

            - ``data`` : list of numpy.ndarray or ``None``  
            The HDU data arrays from each extension.
            - ``header`` : list of astropy.io.fits.Header  
            The corresponding headers.

        Notes
        -----
        ``fits.verify('fix')`` is applied to correct minor FITS standard issues.
        """
        with fits.open(data_path, 'readonly') as file:
            file.verify('fix')
            data = [f.data for f in file]
            header = [f.header for f in file]

        return data, header
        

    def _open_hdf5(self, data_path):
        """
        Open an HDF5 file and recursively convert its contents to a Python dict.

        Parameters
        ----------
        data_path : str or pathlib.Path
            Path to the HDF5 file.

        Returns
        -------
        dict or tuple
            If the root or any group contains attributes under the name
            ``"header"``:

            - ``(data_dict, header_dict)``

            Otherwise only:

            - ``data_dict``

            where ``data_dict`` is a nested dictionary mapping dataset and group
            names to NumPy arrays or sub-dictionaries.
        """
        with h5py.File(data_path, 'r') as file:
            result = self._get_hdf5_dict(file)

        if 'header' in result.keys():
            header = result.pop('header')
            return result, header

        return result
        

    def _get_hdf5_dict(self, file):
        """
        Recursively convert an HDF5 file or group into a nested dictionary.

        Parameters
        ----------
        file : h5py.File or h5py.Group
            An open HDF5 object to inspect.

        Returns
        -------
        dict
            A nested dictionary where each HDF5 group becomes a dictionary and
            datasets become NumPy arrays. Attributes are added either under:

            - ``'header'`` (for file/global or group-level attributes), or  
            - ``'<dataset>_attrs'`` for dataset attributes.
        """
        h5dict = {}
            
        if hasattr(file, 'attrs') and len(file.attrs) > 0:
            h5dict['header'] = dict(file.attrs)

        for key, item in file.items():
            if isinstance(item, h5py.Group):
                if key.lower() == 'header':
                    h5dict = h5dict | self._get_hdf5_dict(item)
                else:
                    h5dict[key] = self._get_hdf5_dict(item)

            elif isinstance(item, h5py.Dataset):
                h5dict[key] = item[()]

                if len(item.attrs) > 0:
                    h5dict[key + '_attrs'] = dict(item.attrs)

        return h5dict
        

    def _open_pickle_data(self, data_path, data_type=np.ndarray):
        """
        Open a pickle file and ensure the returned object has the expected type.

        Parameters
        ----------
        data_path : str or pathlib.Path
            Path to the pickle file.

        data_type : type or tuple of types, optional
            Expected type of the unpickled object.  
            If the object is not an instance of ``data_type``, a ``TypeError`` is
            raised. Default is ``numpy.ndarray``.

        Returns
        -------
        object
            The unpickled Python object.

        Raises
        ------
        TypeError
            If the loaded object is not an instance of ``data_type``.
        """
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            
        if not isinstance(data, data_type):
            raise TypeError(
                f"Pickle object is type {type(data)}, expected {data_type}"
            )

        return data
