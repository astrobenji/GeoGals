'''
cut_data.py

Cut Hii dataframes to inner and outer annulus

Created by: Benjamin Metha
Last updated: Mar 05, 2025
'''

import GeoGals as gg

#diags = ['O3N2', 'N2S2Ha', 'Scal']

metadata = gg.open_metadata()

for meta in metadata:
	gal_ID = meta['Galaxy_ID']
	full_Hii_df = gg.open_Hii_df(gal_ID)
	r_list = gg.RA_DEC_to_radius(full_Hii_df['RA'], full_Hii_df['DEC'], meta)
	r25 = meta['R25_kpc']
	inner_disc_region = (r_list > 0.2*r25) * (r_list < 0.5*r25)
	outer_disc_region = (r_list > 0.5*r25) * (r_list < 0.8*r25)
	inner_Hii_df = full_Hii_df[inner_disc_region]
	outer_Hii_df = full_Hii_df[outer_disc_region]
	inner_Hii_df.to_pickle('../Data/Handmade/Hii_dataframes/inner_disc/Z_maps_{0}.pkl'.format(gal_ID))
	outer_Hii_df.to_pickle('../Data/Handmade/Hii_dataframes/outer_disc/Z_maps_{0}.pkl'.format(gal_ID))
	# Tell me some stats
	print('For ' + gal_ID +':')
	print('Inner disc has {0} pixels'.format(sum(inner_disc_region)))
	print('Outer disc has {0} pixels'.format(sum(outer_disc_region)))