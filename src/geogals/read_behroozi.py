'''
read_behroozi.py

A program to read the data files of Behroozi+13 (and Behroozi+13b)
to convert stellar masses of the PHANGS galaxies
to halo masses,
and hence compute SFEs (q: does this correlate with eta?)

Created by: Benjamin Metha
Last Updated: Apr 16, 2025
'''

import numpy as np
import pandas as pd
import GeoGals as gg

metadata = gg.open_metadata()

stellar_to_halo_mass_file_path = '../Data/behroozi+13/smmr/c_smmr_z0.10_red_all_smf_m1p1s1_bolshoi_fullcosmos_ms.dat'

smhm_df = pd.read_csv(stellar_to_halo_mass_file_path, header=None, delimiter=" ")

log_Mh          = smhm_df[0]
log_Mstar_to_Mh = smhm_df[1]
log_Mstar       = log_Mstar_to_Mh + log_Mh

sfe_file_path = '../Data/behroozi+13/sfe/sfe.dat'

sfe_log_Mh = []
log_sfe    = []
with open(sfe_file_path) as file:
    for line in file:
    	if line == '\n':
    		_, new_Mh, new_sfe = lastline.split(' ')
    		log_sfe.append(float(new_sfe))
    		sfe_log_Mh.append(float(new_Mh))
    	lastline = line

PHANGS_sfes = []
PHANGS_mhalo = []

for meta in metadata:
	m_halo = np.interp(meta['log_M_star'], log_Mstar, log_Mh)
	sfe    = 10**np.interp(m_halo, sfe_log_Mh, log_sfe)
	PHANGS_sfes.append(sfe)
	PHANGS_mhalo.append(m_halo)