'''
Process datacubes

The first step is to convert the datacubes into emission line maps


Created by: Benjamin Metha
Last updated: Oct 17, 2025
'''

from MakeSuperCubes import load_datacubes, get_transformations, get_center_from_header

import lzifu

gal_name   = 'UGC05101'
center_pix = [73, 66]
data_path  = '../Data'


class Cube:

class Meta:

class LineMap:

# Open Cube
def open_cube(gal_name, center_pix):
	cubes, headers, names = load_datacubes()
	
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
	
# Make 1D spectrum
def make_spectrum(Cube, Meta):
	return wavelengths, spectrum

# Extract H alpha 
def extract_Ha(Cube, Meta):
	return Ha_map_kpc

## ALL TOGETHER NOW!

def main(gal_name,center_pix):
    try:
    	open_cube(gal_name, center_pix)
 		print("Processing complete!")
        
    except Exception as e:
        print("Error in processing cube {0}".format(gal_name))
        return False
	
	
if __name__ == '__main__':
	main(gal_name,center_pix)
	