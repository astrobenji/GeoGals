import pickle
import numpy as np
import datetime
from pathlib import Path
from geogals_log import *
from astropy.io import fits
import h5py
import matplotlib.pyplot as plt

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
|   |   └── subrun_XXXXXXXX          # Subrun specific directory (related to analysis?)
|   |                                # with subrun id *
│   │
│   └── log.log                      # Log file generated for this galaxy

To be updated as more directory structure added.

* Runs and subruns are given a unique identifier which is hashed from the parameters which
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
        # self.runs = self._run_scan()

    def _run_scan(self):
        """
        Scan the results directory and build a mapping of run identifiers to
        their parameter dictionaries.

        Returns
        -------
        dict
            Mapping of ``run_id`` (str) -> parameters (dict).
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

        Scalars are duplicated to form length-2 arrays. For example, ``box_size=10``
        becomes ``box_size_x=10`` and ``box_size_y=10``.

        Parameters
        ----------
        **params
            Arbitrary parameter keyword arguments.

        Returns
        -------
        dict
            Parameter dictionary where each parameter ``p`` is replaced by
            ``p_x`` and ``p_y`` with numeric values.

        Raises
        ------
        AssertionError
            If any parameter expands to more than two elements.
        """
        new_parameters = {}
        for key, value in params.items():
            param = np.array([value]).flatten()
            if len(param) == 1:
                param = np.array([param[0], param[0]])
            assert len(param) == 2, "Too many dimensions supplied."
            new_parameters[key + '_x'] = param[0].item()
            new_parameters[key + '_y'] = param[1].item()
        return new_parameters

    def params_to_run(self, simulation=False, n_matches=2, **search_params):
        """
        Find run identifiers that match provided parameter values.

        Parameters
        ----------
        simulation : bool, optional
            If True, convert ``search_params`` to x/y pairs using
            :meth:`_convert_sim_params` and double ``n_matches``. Default False.
        n_matches : int, optional
            Minimum number of parameter matches required. Default 2.
        **search_params
            Parameter key/value pairs to match against stored runs.

        Returns
        -------
        str or list
            If exactly one run matches, return its ``run_id`` as a string.
            Otherwise, return a list of matching ``run_id`` strings.
        """
        runs_found = []
        if simulation:
            search_params = self._convert_sim_params(**search_params)
            n_matches *= 2

        for run_id, run_params in self.runs.items():
            match_params = set(search_params.items()) & set(run_params.items())
            if len(match_params) >= n_matches:
                runs_found.append(run_id)

        if len(runs_found) == 1:
            return runs_found[0]
        return runs_found


class FileHandler:
    """
    Filesystem utilities for creating run directories and reading/writing
    pickled or other data files.

    All methods by Tree.

    Parameters
    ----------
    galaxy_name : str
        Galaxy name used to derive the results directory.
    results_base_directory : str, optional
        Root directory for results. Default './results'.

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
            Base directory in which the galaxy directory will be created.

        Returns
        -------
        pathlib.Path
            The created or existing results directory path.
        """
        results_directory = Path(results_base_directory) / self.galaxy_name
        results_directory.mkdir(parents=True, exist_ok=True)
        return results_directory

    def create_run_directory(self, run_id, subrun_id = None):
        """
        Create a directory for a specific run.

        Parameters
        ----------
        run_id : str
            Run identifier. Directory name: ``run_{run_id}``.

        Returns
        -------
        pathlib.Path
            Path to the run directory.
        """
        run_directory = self._get_run_directory(run_id, subrun_id)
        run_directory.mkdir(exist_ok=True)
        return run_directory

    def _get_run_directory(self, run_id,  subrun_id= None):
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
        if subrun_id is None:
            return self.results_directory / f'run_{run_id}'
        else:
            run_directory = self._get_run_directory(run_id)

            return run_directory / f'subrun_{subrun_id}'

    def load_pickle(self, fname, run_id, log=True):
        """
        Load a pickled object from a run directory.

        Parameters
        ----------
        fname : str
            Filename without extension.
        run_id : str
            Run identifier.
        log : bool, optional
            If True, call object's logger. Default True.

        Returns
        -------
        object
            The unpickled object.
        """
        path = self.results_directory / f'run_{run_id}' / f'{fname}.pkl'
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if log:
            obj.logger.log_read(obj.run_id, obj._fname, 'pickle')
        return obj

    def pickle_object(self, obj):
        """
        Pickle an object into its run directory.

        Parameters
        ----------
        obj : object
            Object must have ``run_id``, ``_fname``, and ``logger`` attributes.

        Returns
        -------
        object
            The same object passed in.
        """
        path = self.results_directory / f'run_{obj.run_id}' / f'{obj._fname}.pkl'
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        obj.logger.log_save(obj.run_id, obj._fname, 'pickle')
        return obj

    def load_data(self, data_path, **kwargs):
        """
        Load a data file using an appropriate loader based on file extension.

        Parameters
        ----------
        data_path : str or pathlib.Path
            Path to the file.
        **kwargs
            Forwarded to the loader.

        Returns
        -------
        object
            Loaded data.
        """
        data_path = Path(data_path)
        loader = self._find_load_method_from_suffix(data_path.suffix.lower())
        return loader(data_path, **kwargs)

    def _find_load_method_from_suffix(self, ext):
        """
        Map file extension to loader method.

        Parameters
        ----------
        ext : str
            File extension including dot (e.g., '.fits').

        Returns
        -------
        callable
            Function to load the file type.

        Raises
        ------
        KeyError
            If unsupported extension.
        """
        if ext in [".fits", ".fit", ".fts"]:
            return self._load_fits
        elif ext in [".h5", ".hdf5"]:
            return self._load_hdf5
        elif ext == ".npy":
            return np.load
        elif ext == ".pkl":
            return self._load_pickle_data
        else:
            raise KeyError(f"Unsupported file extension: {ext!r}")

    def _load_fits(self, data_path, header_only=False, no_header=False, data_names=None, **kwargs):
        """
        Load a FITS file and return its data and headers.

        Parameters
        ----------
        data_path : str or pathlib.Path
            Path to the FITS file.
        header_only : bool, optional
            If True, only return headers.
        no_header : bool, optional
            If True, return only data.
        data_names : list or None, optional
            Names to assign to each HDU data array. Default None.

        Returns
        -------
        tuple or dict
            Tuple ``(data, header)`` or ``data``/``header`` depending on flags.
        """
        data = {}
        with fits.open(data_path, 'readonly') as file:
            file.verify('fix')
            if data_names is None:
                data_names = np.arange(len(file))
            assert len(data_names) == len(file)
            for i, f in enumerate(file):
                if header_only or not no_header:
                    h = dict(f.header)
                    if i == 0:
                        header = h.copy()
                    else:
                        for key, val in h.items():
                            if key in header and val != header[key]:
                                header[f"{key}_{data_names[i]}"] = val
                            else:
                                header[key] = val
                if not header_only:
                    data[data_names[i]] = f.data
            if header_only:
                return header
            if no_header:
                return data
            return data, header

    def load_header_from_file(self, data_path, **kwargs):
        """
        Load only the header from a FITS/HDF5 file.

        Parameters
        ----------
        data_path : str or pathlib.Path
            Path to the data file.

        Returns
        -------
        dict
            Header dictionary.
        """
        return self.load_data(data_path, header_only=True, **kwargs)

    def _load_hdf5(self, data_path, group_name=None, ignore_header=False, header_only=False, **kwargs):
        """
        Load an HDF5 file and convert to nested dictionaries.

        Parameters
        ----------
        data_path : str or pathlib.Path
            Path to the HDF5 file.
        group_name : str, list, optional
            Specific group(s) to extract. Default None.
        ignore_header : bool, optional
            If True, ignore headers. Default False.
        header_only : bool, optional
            If True, return only headers.

        Returns
        -------
        dict or tuple
            Nested dictionary of datasets, optionally with header.
        """
        with h5py.File(data_path, 'r') as file:
            if header_only:
                return self._get_hdf5_header_dict(file)
            result = self._get_hdf5_dict(file, group_name)

        if group_name is not None:
            group_name = np.array([group_name]).flatten()
            if len(group_name) == 1:
                data = result[group_name[0]]
            else:
                data = {name.item(): result[name] for name in group_name}
        else:
            data = result

        if 'header' in result and not ignore_header:
            header = result.pop('header')
            return data, header
        return data

    def _get_hdf5_dict(self, file, group_name=None):
        """
        Recursively convert HDF5 group/file into nested dictionary.

        Parameters
        ----------
        file : h5py.File or h5py.Group
            HDF5 object to convert.
        group_name : str, list, optional
            Specific group(s) to include. Default None.

        Returns
        -------
        dict
            Nested dictionary of data and attributes.
        """
        h5dict = {}
        if hasattr(file, 'attrs') and len(file.attrs) > 0:
            h5dict['header'] = dict(file.attrs)

        for key, item in file.items():
            if isinstance(item, h5py.Group):
                if key.lower() == 'header':
                    h5dict |= self._get_hdf5_dict(item)
                elif group_name is None or key in group_name:
                    h5dict[key] = self._get_hdf5_dict(item)
            elif isinstance(item, h5py.Dataset):
                h5dict[key] = item[()]
                if len(item.attrs) > 0:
                    h5dict[f"{key}_attrs"] = dict(item.attrs)
        return h5dict

    def _get_hdf5_header_dict(self, file):
        """
        Extract only the header information from an HDF5 file.

        Parameters
        ----------
        file : h5py.File or h5py.Group
            HDF5 object.

        Returns
        -------
        dict
            Header dictionary.
        """
        header_dict = dict(file.attrs) if hasattr(file, 'attrs') else {}
        for key, item in file.items():
            if key.lower() == 'header':
                if isinstance(item, h5py.Group):
                    header_dict |= self._get_hdf5_dict(item)['header']
                elif isinstance(item, h5py.Dataset):
                    header_dict[key] = item[()]
        return header_dict

    def _load_pickle_data(self, data_path, data_type=np.ndarray):
        """
        Load a pickle file and validate its type.

        Parameters
        ----------
        data_path : str or pathlib.Path
            Path to the pickle file.
        data_type : type or tuple, optional
            Expected type. Default ``np.ndarray``.

        Returns
        -------
        object
            The loaded pickle object.

        Raises
        ------
        TypeError
            If object is not of expected type.
        """
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, data_type):
            raise TypeError(f"Pickle object is type {type(data)}, expected {data_type}")
        return data

    def save_plot(self, filename, run_id):
        """
        Save the current Matplotlib figure to the run directory.

        Parameters
        ----------
        filename : str
            Name of the file to save.
        run_id : str
            Run identifier.
        """
        filepath = self._get_run_directory(run_id) / filename
        plt.savefig(filepath)
