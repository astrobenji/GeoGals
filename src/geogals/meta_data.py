# by Tree

# for storing and handling metadata
# simulation meta (resolution, physical size and pixel information for projecting)
# last update June 25

import numpy as np
from abc import ABC, abstractmethod
from hyperleda_table import *


class Meta(ABC):
    '''
    Parameters will depend on subclass

    Methods
    -------
    read_meta_from_dicitonary

    read_meta_from_keywords
    '''


    @abstractmethod
    def read_meta_from_dictionary(self):
        # overridden by child class
        pass
    
    def read_meta_from_keywords(self, **kwargs):
        '''
        Construct using keyword arguments, required arguments are the same as the required dict keys for child class read_from_dictionary method

        '''
        return self.read_meta_from_dictionary(kwargs)
    




class SimulationMeta(Meta):

    '''

    Attributes
    ----------
    box_size : [float, float]
        The physical size for each dimension (Length_x, Length_y = box_size)
        if supplied the units are described by distance_units attribute
    N_px : [int, int]
        The number of pixels for each dimension (N_x, N_y = N_px)
    resolution : [float, float]
        The physical size of each pixel for each dimension (resolution_x, resolution_y = resolution)
        if the distance_units attribute isn't None, resolution has units of distance_units/px
    
    Optional Attributes (if included in dictionary or **kwargs)
    -----------------------------------------------------------
    gal_name : str
        The name of the galaxy
    
    distance_units : str
        Units for distances (usually kpc)


    Methods
    -------
    read_meta_from_dictionary



    '''
    
    def read_meta_from_dictionary(self, input_dict):
        '''Construct meta from a dictionary
        
        Parameters
        ----------
        input_dict : dict
            Dictionary containing at least two (2) of,
                - N_px : int or [int, int]
                    The number of pixels
                        * if int, the number of pixels for both dimensions (N_x = N_y = N_px)
                        * if [int, int], the number of pixels in each dimension (N_x, N_y = N_px)

                - box_size : float or [float, float]
                    The physical size of the image
                        * if float, the physical size of the grid in both dimensions so that the 
                        physical shape of the final grid is square (Length_x = Length_y = box_size)
                        * if [float, float], the physical size if the grid for each dimension, so that
                        the physical shape of the final grid is rectangular (Length_x, Length_y = box_size)

                - resolution : float or [float, float]
                    The physical size of each pixel (usually kpc/px),
                        * if float, the size of the pixel in both dimensions so that pixels are square
                        (resolution_x = resolution_y = resolution)
                        * if [float, float], the size of the pixel for each dimension, so that
                        pixels are rectangular (resolution_x, resolution_y = resolution)
                
            May also contain:
                - gal_name : str
                    The name of the galaxy
                - distance_units : str
                    Units for distances (usually kpc)

        

        '''

        # whether variable is in diction
        N_px = 'N_px' in input_dict.keys()
        box_size = 'box_size' in input_dict.keys()
        resolution = 'resolution' in input_dict.keys()

        if N_px:

            # make sure N_px is integers and set attribute
            self.format_parameters(input_dict['N_px'], 'N_px')

            if box_size:
            # set box_size attribute (will override inconsistent resolution):
                self.format_parameters(input_dict['box_size'], 'box_size')

                # calculate resolution and set resolution attribute
                res = self.resolution_from_box_Npx(box_size= self.box_size, N_px=self.N_px)

                self.format_parameters(res, 'resolution')


                if resolution:
                    # do a consistency check with resolution and give warning if required
                    if not self.check_resolution_consistency(input_dict['resolution'], res):
                        print('Warning: Supplied values for box_size, N_px and resolution. Supplied resolution is inconsistent with the box_size and N_px.' \
                        '\nOverriding resolution with value calculated from supplied box_size and N_px.')
            
            elif resolution:
                self.format_parameters(input_dict['resolution'], 'resolution')
                bs = self.box_size_from_N_px_res(N_px=self.N_px, resolution=self.resolution)
                self.format_parameters(bs, 'box_size')


        elif box_size and resolution:

            # set box_size attribute:
            self.format_parameters(input_dict['box_size'], 'box_size')

            #format resolution (don't set yet need to ensure N_px is an integer and recalculate to ensure consistency)
            formatted_input_resolution = self.format_parameters(input_dict['resolution'])


            N = self.N_px_from_box_res(self.box_size, resolution=formatted_input_resolution)

            self.format_parameters(N, 'N_px')

            calculated_resolution = self.resolution_from_box_Npx(box_size=self.box_size, N_px=N)


            if not self.check_resolution_consistency(formatted_input_resolution, calculated_resolution):
                print('To ensure an integer number of pixels, recalculating resolution to be consistent with rounded values.')

            self.format_parameters(calculated_resolution, 'resolution')
        if N_px+box_size+resolution <2:
            raise KeyError('input_dict must contain at least two of N_px, box_size or resolution')
        

        if 'gal_name' in input_dict.keys():
            setattr(self, 'gal_name',  input_dict['gal_name'])

        if 'distance_units' in input_dict.keys():
            setattr(self, 'distance_units',  input_dict['distance_units'])

        return self
    
    def resolution_from_box_Npx(self, box_size, N_px):
        '''Find the resolution from the physical size and number of pixels
        
        Parameters
        ----------
        box_size : [float, float]
            The physical size of the image in each dimension
        
        N_px : [int, int]
            The number of pixels in each dimension

        Returns
        -------
        resolution: [float, float]
            The physical size of a pixel in each dimension

        '''

        return box_size/N_px
    
    def box_size_from_N_px_res(self, N_px, resolution):
        '''Find the physical size of the image from the number of pixels and resolution
        
        Parameters
        ----------
        
        N_px : [int, int]
            The number of pixels in each dimension

        resolution: [float, float]
            The physical size of a pixel in each dimension

        Returns
        -------
        box_size : [float, float]
            The physical size of the image in each dimension

        '''
        return resolution * N_px
    
    def N_px_from_box_res(self, box_size, resolution):
        '''Find the number of pixels from the physical size of the image and resolution
        
        Parameters
        ----------

        box_size : [float, float]
            The physical size of the image in each dimension

        resolution: [float, float]
            The physical size of a pixel in each dimension

        Returns
        -------
        N_px : [int, int]
            The number of pixels in each dimension

        '''
        return (box_size/resolution).astype(int)
    


    def format_parameters(self,param, param_name=None):
        ''' Format parameters to explicitly state along each dimension
        if parameter name provided, will also set attribute
        if the parameter name is N_px, will ensure integer type by rounding up

        Parameters
        ----------

        param : int, float or [int, int], [float, float]
            - if int or float, value will be used for both dimensions
            - if arraylike, must provide one value for each dimension (param_x, param_y = param)

        
        param_name : str
            if provided, attribute will be set using param_name, default is None
            if param_name is 'N_px', will ensure that param is [int, int]

        Returns
        -------
        param : [int, int] or [float, float]
            The value for each dimension (param_x, param_y = param)

        
        
        '''
        param = np.array([param]).flatten()
        if len(param) ==1:
            param = np.array([param[0], param[0]])

        assert len(param) ==2, f'too many dimensions supplied'

        
        if param_name is not None:
            if param_name == 'N_px':
                param = np.ceil(param).astype(int)
            setattr(self, param_name, param)
        
        else:
            return param

    def check_resolution_consistency(self, input_resolution, calculated_resolution):
        ''' 
        Check inputed resolution matches calculated resolution in each dimension

        Parameters
        ----------
        input_resolution : float or [float, float]
            The input resolution
        
        calculated_resolution : [float, float]
            The resolution calculated from box_size and N_px
        '''

        input_resolution = self.format_parameters(input_resolution, param_name=None)

        # return False if either input disagrees:
        agreement = np.prod([input_resolution[i] == calculated_resolution[i] for i in range(2)]).astype(bool)
        return agreement


SimMeta = SimulationMeta()


class GalaxyMeta(Meta):
    '''
    Galaxy Meta
    '''
    def read_meta_from_dictionary(self, meta_dict, info_dict=None):
        '''
        read galaxy meta from dictionary

        Parameters
        ----------
        meta_dict : dict
            dictionary of parameters
        
        info_dict : dict (optional)
            dictionary of info about the parameters (citations, units, uncertainties etc)

        '''

        for key, value in meta_dict.items():
            setattr(self,key, value)

        
        if info_dict is not None:
            setattr(self, 'info', info_dict)
    
    def get_paid(self, gal_name, q=0.2):
        '''
        use hyperleda to get meta
        '''
        table = HyperLedaTable(gal_name)
        search = {'ra_dec': ['RA', 'DEC'], 'main': ['pa', 'incl', 'modbest', 'logr25']}
        for search_type, params in search.items():
            table.get_table(search_type)
        
            table.find_in_table(params)

        param_dict, info_dict = table.parameters_dict()

        if 'pa' not in param_dict.keys():
            table.get_table('pa')

            table.find_in_table(['pa'])
            
            param_dict, info_dict = table.parameters_dict()
        
        if 'incl' in param_dict.keys() and 'logr25' in param_dict.keys():
            param_dict.pop('logr25')
            info_dict.pop('logr25')
        elif 'incl' not in param_dict.keys():
            if 'logr25' in param_dict.keys():
                param_dict['incl'], info_dict = self.convert_param_dict('incl', ['logr25'], info_dict, lambda _: self.logr25_to_incl(_, q))
                param_dict.pop('logr25')

            else:
                table.get_table('incl', select_criteria=select_criteria)
                table.find_in_table(['lax', 'sax'])
                param_dict, info_dict = table.parameters_dict()
                param_dict['incl'], info_dict = self.convert_param_dict('incl', ['lax', 'sax'], info_dict, lambda _: self.lax_sax_to_incl(_, q))
                param_dict.pop('lax')
                param_dict.pop('sax')
            
        if 'modbest' in param_dict.keys():
            param_dict['distance'], info_dict = self.convert_param_dict('distance', ['modbest'], info_dict, self.modbest_to_distance)
            param_dict.pop('modbest')

        else:
            table.get_table('distance')
            table.find_in_table(['distance'])
            _, new_info_dict = table.parameters_dict()
            param_dict['distance'], new_info_dict = self.convert_param_dict('distance', ['distance'], new_info_dict, lambda x: x)
        param_dict['PA'] = param_dict.pop('pa')
        
        param_dict['i'] = param_dict.pop('incl')
        param_dict['D'] = param_dict.pop('distance')

        self.read_meta_from_dictionary(param_dict, info_dict)

        return self
    
    def logr25_to_incl(self, logr25, q = 0.2):
        ''' convert log of axis ratio ... to inclination
        '''

        # Convert b/a ratioto incl, in degrees, assuming q=0.2
        # based on Battisti+17, Eq. (1):
        # https://ui.adsabs.harvard.edu/abs/2017ApJ...851...90B
        logr25 = logr25[0]
        

        b_on_a = 10**(-logr25)
        incl_rad = np.arccos(np.sqrt((b_on_a**2 - q**2)/(1 - q**2)))
        incl_deg = np.degrees(incl_rad)

        

        return incl_deg

    def lax_sax_to_incl(self, log_axes, q=0.2):
        '''
        convert log of major axis and log of minor axis of the isophote 25 mag/arcsec2 in the  B-band for galaxies to incl
        '''
        lax, sax = log_axes
        logr25 = lax - sax

        return self.logr25_to_incl([logr25], q)

    def convert_param_dict(self, parameter, derived_from, info_dict, convert_func):
        '''Convert info_dict to new value from convert_func'''

        values = [info_dict[param]['Value'] for param in derived_from]

        param_value = convert_func(values)


        unc_source = f'derived from {''.join(derived_from)}'
        info_dict[parameter] = {'Parameter': parameter,
                                'Value': param_value,
                                'Uncertainty': unc_source,
                                'source': unc_source}
        return param_value, info_dict

    def modbest_to_distance(self, modbest):
        '''Convert modbest to distance'''
        modbest = modbest[0]
        D_Mpc     = 10**((modbest - 25)/5)
        return D_Mpc
    
    def to_dict(self, get_info=True):
        meta_dict = self.__dict__.copy()
        info_dict = meta_dict.pop('info')
        if get_info:
            return meta_dict, info_dict
        return meta_dict
            

GalMeta = GalaxyMeta()