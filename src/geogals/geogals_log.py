import numpy as np
import datetime
from pathlib import Path
from geogals_io import *

''' 
By Tree

Module to assist the user in understanding their progress. Logger ensures that
processes are timestamped and categorised for processes including initialising
geogals objects, saving and loading files, loading data from one geogals into 
another and creating runs.

Logging is specific to each galaxy and information is stored in a log.log file
within the galaxy results directory. The log file is produced on a per galaxy
basis and all runs for that galaxy are logged to the same file.


'''

class Logger:
    """
    Logger for timestamped, categorized log entries stored in a galaxy-specific
    results directory.

    All methods by Tree.

    Parameters
    ----------
    galaxy_name : str
        Name of the galaxy used to locate the results directory.
    results_base_directory : str, optional
        Base path under which galaxy result directories are created. Default
        is './results/'.

    Attributes
    ----------
    galaxy_name : str
        As provided.
    results_directory : pathlib.Path
        Path to the galaxy's results directory.
    log_file : pathlib.Path
        Path to the log file inside the results directory.
    """

    def __init__(self, galaxy_name, results_base_directory = './results/'):
        self.galaxy_name = galaxy_name
        self.results_directory = FileHandler(galaxy_name)._create_results_directory(results_base_directory)
        self.log_file = self.results_directory / 'log.log'
        if not self.log_file.is_file():
            self.log('INIT','Log created.')


    def log(self, log_code, msg):
        """
        Append a single log line to the log file.

        Parameters
        ----------
        log_code : str
            Short code categorising the log entry (for example 'INIT', 'RUN',
            'SAVE', 'LOAD', 'PARAM').
        msg : str
            Human-readable message to record.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"{timestamp} | {log_code} | {msg}\n")

    def log_run(self, run_id):
        """
        Record that a new run has been created.

        Parameters
        ----------
        run_id : str
            Unique identifier of the run.
        """
        self.log('RUN', f'(run id: {run_id}) Run created.')

    def log_init(self, obj_name, run_id=None):
        """
        Record that an object has been initialised.

        Parameters
        ----------
        obj_name : str
            Name of the object being initialised (e.g. 'meta').
        run_id : str, optional
            If provided, include the associated run identifier in the log
            message.
        """
        msg = f'{obj_name.capitalize()} initialised.'
        if run_id is not None:
            msg = f'(run id: {run_id}) {msg}'
        
        self.log('INIT', msg)



    def log_save(self, run_id, save_type, file_type):
        """
        Record that an object has been saved to disk.

        Parameters
        ----------
        run_id : str
            Identifier of the run associated with the saved object.
        save_type : str
            Semantic type of object saved (for example, 'meta', 'data').
        file_type : str
            File format used (for example, 'pickle').
        """
        
        self.log('SAVE', f'(run id: {run_id}) {save_type.capitalize()} saved to {file_type}.')

    def log_read(self, run_id, data_type, file_type):
        """
        Record that an object has been read from save.

        Parameters
        ----------
        run_id : str
            Identifier of the run associated with the loaded object.
        data_type : str
            Semantic type of object loaded (for example, 'meta', 'data').
        file_type : str
            File format used (for example, 'pickle').
        """
        self.log('READ', f'(run id: {run_id}) {data_type.capitalize()} read from {file_type}.')
        pass

    def log_load(self, run_id, load_type, load_into):
        """
        Record that an object has been loaded from disk or loaded into another
        object.

        Parameters
        ----------
        run_id : str
            Identifier of the run associated with the loaded object.
        load_type : str
            Semantic type of object loaded (for example, 'meta', 'data').
        load_into : str
            Records the target object into which the data was loaded.
        """
        self.log('LOAD', f'(run id: {run_id}) {load_type} loaded into {load_into.capitalize()}.')

    def log_params(self, run_id, params_dict):
        """
        Record a dictionary of parameters associated with a run.

        Parameters
        ----------
        run_id : str
            Identifier of the run.
        params_dict : dict
            Mapping of parameter names to values. Values will be formatted as
            ``key = value`` pairs in the log entry.
        """
        params_str = ', '.join([f'{key} = {val}' for key, val in params_dict.items()])

        msg = f'(run id: {run_id}) {params_str}'
        self.log('PARAM', msg)
    
    def log_load_attrs(self, run_id,load_into, attr_name, attr_value, dict_name = None):
        msg = f'{attr_name} stored in {load_into}'

        
        if run_id is not None:
            msg = f'(run id: {run_id}) ' + msg


        if isinstance(attr_value, dict):
            self.log('PARAM', msg)
            for key, value in attr_value.items():
                self.log_load_attrs(run_id, load_into, key, value, dict_name=attr_name)
        

        else:
            

            if dict_name is not None:
                msg += f' within {dict_name}'
            if not hasattr(attr_value, '__iter__'):
                msg +=  f' ({attr_name} = {str(attr_value)})'
            self.log('PARAM', msg)

