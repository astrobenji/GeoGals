'''
Process datacubes

The first step is to convert the datacubes into spectra.

The second step is to get emission line maps.

The third step is to convert the emission line maps into feature maps.

Created by: Benjamin Metha
Last updated: Nov 14, 2025
'''

from MakeSuperCubes import load_datacubes, get_transformations, get_center_from_header

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

################################
#                              #
#        Hypercube/Cubes       #
#                              #
################################

class Cube:

# Open Cube
def open_cube(gal_name, center_pix, path):
	'''
	Wrapper for hypercube
	
	Parameters
	----------
	
	gal_name: str
	
	center_pix: list or tuple of 2 numbers
	
	path: str
		Name of the location of the data
	'''
	cubes, headers, names = load_datacubes(path)
	
	# Get user-specified transformations
	transformations = get_transformations(names)
	
	# Get wavelength information
	wave_info = get_wavelength_info(headers)
	for name, info in zip(names, wave_info):
		print(f"{name}: {info['min']:.1f}-{info['max']:.1f} angstrom, step={info['step']:.2f} angstrom")
	
	# Create white light images for alignment
	white_images = create_white_images(cubes)
	
	# Align images with transformations
	aligned_images, shifts, max_shape = align_images(white_images, names, transformations)
	return Cube, Meta
	
def open_coadd_cubes(gal_name, center_pix, path):
	extensions = ['BH1L', 'RH1L', 'BLL', 'RH2L']
	hypercube = {}
	for ext in extensions:
		HDU = fits.open(path + gal_name + '_coadd_' + ext + '.fits')
		hypercube[ext] = HDU[0]
	return hypercube 

def crop_hypercube(hypercube, target_areas):
	cropped_cubes = dict()
	for c in hypercube.keys():
		xlims   = target_areas[c]['x']
		ylims   = target_areas[c]['y']
		cropped_cubes[c] = hypercube[c].data[:, xlims[0]:xlims[1], ylims[0]:ylims[1]]
	return cropped_cubes

################################
#                              #
#             Maps             #
#                              #
################################


class Maps:
	'''
	Maps of features of galaxies
	'''
	def __init__(hypercube):
		'''
		'''
		
	def trim(target_areas):
	
def make_total_flux_maps(hypercube):
	maps = {} # should be all feature maps though
	for ext in hypercube.keys():
		cube = hypercube[ext]
		flux_map = cube.data.sum(axis=0)
		maps[ext] = flux_map
	return maps
	
# Extract H alpha 
def extract_Ha(hypercube):
	return maps
	
def crop_maps(maps, target_areas):
	for c in maps.keys():
		xlims   = target_areas[c]['x']
		ylims   = target_areas[c]['y']
		maps[c] = maps[c][xlims[0]:xlims[1], ylims[0]:ylims[1]]
	return maps
	
def make_metallicity_and_error_maps(maps, diagnostics):
	return maps

################################
#                              #
#       Spectrum/Spectra       #
#                              #
################################	

class Spectrum:
	'''
	Attributes:
	
	Wavelengths -- 1D Numpy array
	Flux
	
	length (n)
	max_wavelength
	min_wavelength
	
	limits -- 2-tuple
	'''
		
def make_spectra(hypercube, target_areas = None):
	spectra = {} # Spectra.init
	for c in hypercube.keys():
		cube       = hypercube[c]
		spectra[c] = make_spectrum(cube, target_areas)
	return spectra
	
def make_spectrum(cube, target_areas = None):
	# Read wavelengths
	min_lambda  = cube.header['WAVALL0']
	max_lambda  = cube.header['WAVALL1']
	n_lambdas   = cube.header['NAXIS3'] 
	wavelengths = np.linspace(min_lambda, max_lambda, n_lambdas)
	# Sum total flux
	if target_areas is None:
		spectrum = cube.data.sum(axis=(1,2))
	else:
		xlims   = target_areas[c]['x']
		ylims   = target_areas[c]['y']
		cropped_cube = cube.data[:, xlims[0]: xlims[1], ylims[0]: ylims[1]]
		spectrum = cropped_cube.sum(axis=(1,2))
	return [wavelengths, spectrum]
	
def plot_label_lines(spectrum, lines):
	'''
	lines: dict
	List of lines and their observed predicted wavelengths
	'''
	wavelength = spectrum[0]
	
def trim_spectra(spectra):
	'''
	Change the limits of the spectrum object
	'''
	# Read header for wavelength limits		

################################
#                              #
#        Galaxy specifics      #
#                              #
################################	

gal_name   = 'UGC05101'
center_pix = [73, 66]
data_path  = '../Data/cubes/'

configurations = ['BH1L', 'BLL', 'RH1L', 'RH2L']

colours = { 'BH1L': '#92c5de', 
             'BLL': '#0571b0', 
            'RH1L': '#ca0020', 
            'RH2L': '#f4a582'}

colormaps = {'BH1L': cmr.voltage, 
             'BLL' : cmr.arctic, 
             'RH1L': cmr.ember, 
             'RH2L': cmr.amber}
             
target_areas = {'BH1L': {'y':[67,98], 'x':[53,63]}, 
                'BLL' : {'y':[30,57], 'x':[77,87]}, 
                'RH1L': {'y':[30,57], 'x':[77,87]}, 
                'RH2L': {'y':[53,87], 'x':[53,63]}}

def main(gal_name,center_pix):
    try:
    	#open_cube(gal_name, center_pix, data_path)
    	hypercube = open_coadd_cubes(gal_name, center_pix, data_path, target_areas)
    	spectra   = make_spectra(hypercube, target_areas)
    	maps      = make_total_flux_maps(hypercube)
 		print("Processing complete!")
        
    except Exception as e:
        print("Error in processing cube {0}".format(gal_name))
        return False
    
    
    # Open all spectra
#     BH1L = make_spectrum(cropped_cubes['BH1L'])
#     BLL  = make_spectrum(cropped_cubes['BLL'])
#     RH1L = make_spectrum(cropped_cubes['RH1L'])
#     RH2L = make_spectrum(cropped_cubes['RH2L'])  
	
	
if __name__ == '__main__':
	main(gal_name, center_pix, data_path, target_areas)
	