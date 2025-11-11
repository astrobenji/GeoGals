'''

Tests the ability to open a datacube and make 1D plots

Created by: Benjamin Metha
Last updated: Nov 11, 2025
'''

from Process_Datacubes import *

def test_open_cube():
	test_gal        = 'UGC05101'
	test_center_pix = [73, 66]
	try:
		open_cube(test_gal, test_center_pix):
	except:
		return False
	
def test_create_white_images():
	test_gal        = 'UGC05101'
	test_center_pix = [73, 66]
	try:
		hypercube = open_cube(test_gal, test_center_pix)
		make_total_flux_maps(hypercube)
		return True
	except:
		return False
		
def test_get_spectrum():
	test_gal        = 'UGC05101'
	test_center_pix = [73, 66]
	try:
		hypercube = open_cube(test_gal, test_center_pix)
		make_spectra(hypercube)
		return True
	except:
		return False