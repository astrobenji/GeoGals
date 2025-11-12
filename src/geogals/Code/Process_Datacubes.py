'''
Process datacubes

The first step is to convert the datacubes into emission line maps.

The second step is to convert the emission line maps into feature maps


Created by: Benjamin Metha
Last updated: Nov 11, 2025
'''

from MakeSuperCubes import load_datacubes, get_transformations, get_center_from_header

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# import lzifu

gal_name   = 'UGC05101'
center_pix = [73, 66]
data_path  = '../Data/cubes/'


class Cube: ...

class Meta: ...

class LineMap: ...

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
	...

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

# Make 1D spectrum for all configurations
def make_spectra(hypercube):
	spectra = {} # Spectra.init
	for ext in hypercube.keys():
		cube = hypercube[ext][0]
		# Read wavelengths
		min_lambda  = cube.header['WAVALL0']
		max_lambda  = cube.header['WAVALL1']
		n_lambdas   = cube.header['NAXIS3']
		wavelengths = np.linspace(min_lambda, max_lambda, n_lambdas)
		# Sum total flux
		spectrum     = cube.data.sum(axis=(1,2))
		spectra[ext] = [wavelengths, spectrum]
	return spectra

def plot_spectra(spectra):
	n = len(spectra)
	fig, ax = plt.subplots(n)
	for ii, ext in enumerate(spectra.keys()):
		wavelength = spectra[ext][0]
		spectrum   = spectra[ext][1]
		ax[ii].plot(wavelength, spectrum, label=ext)
		ax[ii].set_xlimits(spectrum.limit[0], spectrum.limit[1])
	plt.legend()
	plt.xlabel('Wavelength')
	plt.tight_layout()
	return fig, ax

def trim_spectra(spectra):
	'''
	Change the limits of the spectrum object
	'''
	# Read header for wavelength limits

def make_total_flux_maps(hypercube):
	maps = {} # should be all feature maps though
	for ext in hypercube.keys():
		cube = hypercube[ext][0]
		flux_map = cube.data.sum(axis=0)
		maps[ext] = flux_map
	return maps

def plot_maps(maps):
	fig, ax = plt.subplots(2,2) # Generalise later
	ii = 0
	for ext in enumerate(maps.keys()):
		ax[ii] = plt.imshow(m[ext])
		ax[ii].set_title(ext)
		ii += 1
	return fig, ax



# Extract H alpha
def extract_Ha(Cube, Meta):
	return Ha_map_kpc

## ALL TOGETHER NOW!

def main(gal_name,center_pix):
    try:
    	#open_cube(gal_name, center_pix, data_path)
    	test_gal = open_coadd_cubes(gal_name, center_pix, data_path)
    	print("Processing complete!")

    except Exception as e:
        print("Error in processing cube {0}".format(gal_name))
        return False


if __name__ == '__main__':
	main(gal_name,center_pix)

