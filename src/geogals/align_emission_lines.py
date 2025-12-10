'''
align_emission_lines.py

The third step is to connect emission line properties between different cubes.

'''

import numpy as np
from reproject.mosaicking import find_optimal_celestial_wcs
from astropy.io import fits
from astropy.wc import WCS

from reproject import reproject_interp, reproject_exact
from reproject.mosaicking import reproject_and_coadd

import pickle

def open_cube(gal_ID, config):
	return fits.open(f'../Data/coadded_cubes/{gal_ID}_{config}_coadd.fits')

def open_line_maps(gal_ID, config):
	filename = '../Data/LineMaps/{gal_ID}/{config}_lines.pkl'
	with open(filename, 'rb') as handle:
    	return pickle.load(handle)

# Open coadded cubes (with headers)

gal_ID = 'F23365'

configurations = ['BH1A', 'BH3', 'RH2']

cubes = []
for config in configurations:
	cubes.append(open_cube(gal_ID, config))
	
wcs_out, shape_out = find_optimal_celestial_wcs(cubes)

wcs_list = [WCS(cube[0].header) for cube in cubes]

# Open line maps objects

# Project them all onto the same WCS
AllLineMaps = dict{}
for ii, config in enumerate(configurations):
	LineMaps = open_line_maps(gal_ID, config)
	wcs_in   = wcs_list[ii]
	for p in LineMaps.keys():
		AllLineMaps[p], _ = reproject_and_coadd((LineMaps[p], wcs_in), wcs_out, shape_out=shape_out,
                                       reproject_function=reproject_exact)

# Save WCS in a meta object

with open(save_path + f'{gal_ID}/All_LineMaps.pkl', 'wb') as handle:
    pickle.dump(LineMaps, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# Save LineMaps as a hdu?