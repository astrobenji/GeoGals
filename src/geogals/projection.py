# by Tree

# for creating simple projections from simulated particle data
# last update 15/7/25

import numpy as np
from meta_data import SimulationMeta

import utilities as ut

class Projection(SimulationMeta):
    ''' 
    class for projecting three dimensional particle data into two dimensional images
    '''
    def __init__(self, meta, data_dict, centre=[0,0,0] ,z_bound=np.inf, rotation_angles=None):
        '''
        Parameters
        ----------
        meta : dict or SimulationMeta class
            if meta is a dictionary must contain the following keys
                *  box_size : [float, float]
                    The physical size for each dimension (Length_x, Length_y = box_size)
                *  N_px : [int, int]
                    The number of pixels for each dimension (N_x, N_y = N_px)
                *  resolution : [float, float]
                    The physical size of each pixel for each dimension (resolution_x, resolution_y = resolution)
            (see SimulationMeta for more information)
        
        data_dict : dict
            dictionary containing particle data, must contain the following key(s)
                * position : ndarray shape(N,3)
                    The three dimensional position of each particle in cartesian coordinates
            Special keys(s):
                * metallicity
                    The metal mass fraction of each particle
        
        centre : [float, float], optional
            the centre of the galaxy in two dimensions, default is [0,0]

        z_bound : float, optional
            the maximum distance the galaxy plane, default is np.inf

        rotation_angles : [float, float, float]
            rotation angles about x-axis, y-axis, z-axis (radians)
        

        '''
        self.read_meta(meta) # load resolution information
        self.boundaries = self.find_boundaries() # image boundaries
        self.bin_edges = self.find_pixel_boundaries() # pixel boundaries
        self.included_particles, self.position = self.format_position(data_dict, z_bound, rotation_angles, centre) # rotated positions and whether particle is in bounds
        self.add_particle_data(data_dict) # data dictionary
        self.x_bindices, self.y_bindices = self.bin_positions() # index of pixel particle falls into along each dimension
        self.total_px = np.prod(self.N_px) # total number of pixels in the final image (N_px[0] * N_px[1])

    
    def read_meta(self, meta):
        '''
        read meta data

        Parameters
        ----------
        meta : dict or SimulationMeta class
        if meta is a dictionary must contain the following keys
            *  box_size : [float, float]
                The physical size for each dimension (Length_x, Length_y = box_size)
            *  N_px : [int, int]
                The number of pixels for each dimension (N_x, N_y = N_px)
            *  resolution : [float, float]
                The physical size of each pixel for each dimension (resolution_x, resolution_y = resolution)
        (see SimulationMeta for more information)
        '''

        if type(meta) == dict:
            SimulationMeta.read_meta_from_dictionary(self,meta)
        elif type(meta) == SimulationMeta:
            SimulationMeta.read_meta_from_dictionary(self,meta.__dict__)


    def find_boundaries(self):
        '''
        find the boundaries of box with size self.box_size centred on centre

        Parameters
        ----------
        None


        Returns
        -------
        boundaries : ndarray, shape(2,2)
            The leftmost and rightmost edges of the bins along each dimension

        '''
        
        bound_vals = self.box_size/2

        


        return np.vstack((-bound_vals, bound_vals)).T
    
    def find_pixel_boundaries(self):
        '''
        Find the edges of the pixels in each dimension 

        Returns
        -------
        pixel_boundaries : [array, array]
            the pixel edges in each dimension (x_edges, y_edges = pixel_boundaries)
        
        '''

        return [np.linspace(self.boundaries[i][0], self.boundaries[i][1], self.N_px[i]+1) for i in range(2)]
    
    def format_position(self, data_dict, z_bound, rotation_angles, centre):
        '''
        rotate positions and reduce data

        Parameters
        ----------
        data_dict : dict

        z_bound : float

        rotation_angles : [float, float, float]

        see contructor for details of each parameter

        Returns
        -------
        included_particles : ndarray, shape (N,)
            Whether or not each particle falls within the bounds

        pos : ndarray, shape(M,3)
            the rotated three dimensional positions of particles
            the length should be the same as sum(included_particles)
        '''
        assert 'position' in data_dict.keys()

        centre = np.array(centre)


        pos = data_dict['position'] + centre

        


        if rotation_angles is not None:

            pos = self.rotate(pos, rotation_angles)

        included_particles = self.ids_in_bounds(pos, z_bound)





    

        return included_particles, pos[included_particles]

    
    def ids_in_bounds(self, positions, z_bound = np.inf):
        '''
        find whether each particle position falls in the boundaries
        
        Parameter
        --------
        positions : array_like, shape(N,3)
            three dimensional particle positions
        
        z_bound : float
            see constructor for details
        
        '''
        included_particles = np.prod([(positions[:,i] > self.boundaries[i][0]) * (positions[:,i]< self.boundaries[i][1]) for i in range(2)], axis=0)
        included_particles *= (positions[:,2] > -z_bound) * (positions[:,2] < z_bound)
        return included_particles.astype(bool)
    

    
    def add_particle_data(self, particle_data):

        ''' 
        Add particle data to the data dictionary, list of particle data should have the same shape as included_particles

        Parameters
        ----------
        particle_data : dict
            a dictionary of particle data to add
            the dictionary values should be one dimensional arrays and should have the same shape as self.included_particles

        Returns
        -------
        data : dict
            the new data dictionary with the new values added
        '''
        data_dict = particle_data.copy()
        if not hasattr(self, 'data'):
            setattr(self, 'data', {})
        
        if 'position' in data_dict:
            data_dict.pop('position')



        for key, val in data_dict.items():
            self.data[key] = val[self.included_particles]

        return self.data
    
    def bin_positions(self):
        ''' 
        Find the bin index (bindex) of each particle along each dimension

        Returns
        -------
        x_bindices, y_bindices : ndarray
            the indices of the bins to which each value in input array belongs along each dimension


        '''
        x,y = (self.position[:,:-1]).T
        
        
        x_bindices = np.digitize(x, self.bin_edges[0]) - 1

        y_bindices = np.digitize(y, self.bin_edges[1]) - 1


        return x_bindices, y_bindices
    

    def find_bin_centres(self, bin_edges=None, dim = None):
        ''' 
        find the coordinate of the centre of each pixel along a dimension

        Parameters
        ----------
        bin_edges : [array] or [array, array], optional
            the edges of the pixels in a single dimension or along each dimension
            default is None, results in self.bin_edges as the value
        
        dim : int
            the dimension along which to find pixel centres, must not be None if bin_edges is two dimensional
            default is None, if bin_edges is also None will raise Exception
        
        Returns
        -------
        bin_centres : array
            the coordinate of the centre of each pixel along a single dimension

        '''
        if bin_edges is None:
            bin_edges = self.bin_edges

        if len(bin_edges) == 2 and dim is None:
            raise Exception('please specify the bin_edges along a single dimension or specify the dimension to find the centres along')
        elif dim is not None:
            bin_edges = bin_edges[dim]

        return 0.5*(bin_edges[1:] + bin_edges[:-1])
    
    def find_pixel_centre_coord_grid(self, bin_edges = None):
        ''' 
        get xy grid of coordinates of the centre of each pixel

        Parameters
        ----------
        bin_edges : [array, array], optional
            the edges of the pixels along each dimension
            default is None, in which case the value of self.bin_edges is used
        
        Returns
        -------
        x_grid, y_grid : array, shape(N_px, N_px)
            the x and y coordinates of the centre of each pixel
        

        '''

        if bin_edges is None:
            bin_edges = self.bin_edges

        assert len(bin_edges) == 2, 'please specify the bin_edges along both dimensions'

        return np.meshgrid(self.find_bin_centres(bin_edges[0]), self.find_bin_centres(bin_edges[1]))
    
    def project_values(self, values, per_unit_area=False):
        '''
        project a list of particle values corresponding to self.position

        Parameters
        ----------
        values : arraylike
            a list of particle values to project
        
        per_unit_area : bool, optional
            when True returned projected values are per pixel area
            default is False

        Returns
        -------
        projected_values : array (N_px, N_px)
            Pixels whos value is sum of the particle values whos position fall within the pixel
        '''
        px_area = 1
        if per_unit_area:
            px_area = np.prod(self.resolution)

        multi_idx = np.ravel_multi_index((self.x_bindices,self.y_bindices), self.N_px)

        return np.bincount(multi_idx, weights=values, minlength=self.total_px).reshape(self.N_px)/px_area
    
    def project(self, var_name, per_unit_area=False):
        '''
        project a value from the data dictionary

        Parameters
        ----------
        var_name : str
            a key from the data_dict whos values will be projected into two dimensions
        
        per_unit_area : bool, optional
            when True returned projected values are per pixel area
            default is False

        Returns
        -------
        projection : array (N_px, N_px)
            Pixels whos value is sum of the particle values whos position fall within the pixel
        '''
        vals = self.data[var_name]
        

        return self.project_values(vals, per_unit_area=per_unit_area)
    
    def weighted_project(self, var_name, weights_function):
        '''
        weighted projection from the particle dictionary

        weight the pixel by the values returned from a function

        Parameters
        ----------
        var_name : str
            a key from the data_dict whos values will be projected into two dimensions
        
        weights_function : callable
            a function that takes as its only argument the data_dict and returns the value of the weights
            to use for the projection

        Returns
        -------
        projection : array (N_px, N_px)
            Pixels whos value is the sum(weighted by the result of weights_function)
            of the particle values whos position fall within the pixel
            

        
        
        
        '''
        vals = self.data[var_name]

        weights = self.get_weights(weights_function)



        return self.project_values(vals * weights)/ self.project_values(weights)
    
    def get_weights(self, weights_function):
        '''
        get the value of the weights

        '''
        return weights_function(self.data)
    
    def metallicity_map(self):
        '''
        Create a metallicity map, final pixel values are in dex relative to solar, using the solar
        metallicity value from Asplund et al. 2009.

        This method is only valid if the data dictionary contains the key 'metallicity' where 
        the values are the total mass fraction of metals for each particle

        Parameters
        ----------
        None

        Returns
        -------
        metallicity_map : array (N_px, N_px)
            Value of each pixel is the metallicity in dex relative to solar
            
        '''

        weight_function = lambda data: data['mass']

        Z = self.weighted_project('metallicity', weight_function)

        return np.log10(Z/0.0134) # metallicity relative to solar


    
    def weighted_metallicity_map(self, weights_function):
        '''
        Create a weighted metallicity map, final pixel values are in dex relative to solar, using the solar
        metallicity value from Asplund et al. 2009.

        Parameters
        ----------
        weights_function : callable
            a function that takes as its only argument the data_dict and returns the value of the weights
            to use for the projection

        Returns
        -------
        metallicity_map : array (N_px, N_px)
            Pixels whos value is the fraction of the sum of the pixel mass which is metals
            (weighted by the result of weights_function)
        
        '''
        vals = self.data['metallicity']
        mass_weights = self.data['mass']
        weights = self.get_weights(weights_function)
        
        projection = self.project_values(vals * weights * mass_weights) / (self.project_values(mass_weights) * self.project_values(weights))


        
        return np.log10(projection/0.0134)
    
    def rotate(self, pos, rotation_angles):
        '''
        rotate coordinates

        uses function from the utilities_awetzel package

        Parameters
        ----------
        rotation_angles : [float, float, float]
            rotation angles about x-axis, y-axis, z-axis (radians)
        '''

        pos = ut.coordinate.get_coordinates_rotated(pos, rotation_angles=rotation_angles)
        return pos
        

        
        
    


        