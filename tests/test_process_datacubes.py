'''

Tests the ability to open a datacube and make 1D plots

Created by: Benjamin Metha
Last updated: Oct 17, 2025
'''

from MAUNA_01_Process_Datacubes import *

def test_open_cube():
	test_gal        = 'UGCA515'
	test_center_pix = [73, 66]
	try:
		open_cube(test_gal, test_center_pix):
	except:
		return False
	
def test_create_white_images():
	test_gal        = 'UGCA515'
	test_center_pix = [73, 66]
	try:
		Meta, Cube = open_cube(test_gal, test_center_pix)
		create_white_images(Cube)
		return False
	except:
		return True
		
def test_get_spectrum():
	test_gal        = 'UGCA515'
	test_center_pix = [73, 66]
	try:
		Meta, Cube = open_cube(test_gal, test_center_pix)
		get_spectrum(Cube)
		return False
	except:
		return True