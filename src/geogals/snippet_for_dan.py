import numpy as np # almost goes without saying...

def exp_4D_log_prob_global(theta):
	'''
	Functional form of the log likelihood of the model, with 4 parameters
	Adapted from eq. 5.14-5.16 of Diggle+Riberio07, but we don't fit tau^2;
	hence, knowing sigma is equivalent to knowing nu, so we don't estimate that 
	either.
	
	Uses a Matern function with an index of 1/2, A.K.A. an exponential.
	
	Uses difficult maths that I haven't commented well; look up Cholesky 
	decomposition. It's a fast way to solve big matrix equations.
		
	Parameters
	----------
	theta: 4-tuple
		Contains:
			A, phi: model parameters for spatial_cov
			Z_char, gradZ: model parameters for the large scale gradient
			
	THESE ALL NEED TO BE DEFINED AS GLOBAL VARIABLES:
								 
	Z: (N,) np.array
		Observations at N data points
	
	r: (N,) np.array
		Covariate that is each spaxel's distance from the galaxy center.
				
	e_Z: (N,N) np.array
		matrix of observation variance at all data points. 
		
	dist_matrix: (N,N) np.array
		matrix of distance between all observed data points.
		
	Returns
	-------
	log_likelihood: of this model.
	'''
	log_A, phi, Z_char, gradZ = theta
	A = 10**log_A
	# infold a prior on phi really quick:
	if phi < 0.05:
		return -np.inf
	if phi > 3:
		return -np.inf
	# Exponential covariance
	r_on_phi  = dist_matrix / phi
	spatial_cov = A* np.exp( -1.0 * r_on_phi)
	# using notation from Diggle now, D is covariates
	# V is variance
	V = e_Z + spatial_cov
	L = np.linalg.cholesky(V)
	log_det_L = np.sum(np.log(np.diag(L)))
	log_det_V = 2*log_det_L
	white_D = np.linalg.solve(L, D)
	white_Z = np.linalg.solve(L, Z)
	# subtract mean trend
	beta = np.array([Z_char, gradZ])
	resids =  Z - char_D @ beta
	white_resids = np.linalg.solve(L, resids)
	chi_sq = n*np.log(2*np.pi) + log_det_V + white_resids.T @ white_resids
	return -0.5*chi_sq
	
