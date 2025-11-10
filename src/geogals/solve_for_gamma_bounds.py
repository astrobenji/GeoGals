import scipy
import numpy as np

def func(x, lower_lim=0.02, upper_lim=0.5):
	alpha = x[0]
	beta  = x[1]
	delta_low  = scipy.special.gammainc(alpha, beta*lower_lim) - 0.01
	delta_high = scipy.special.gammainc(alpha, beta*upper_lim) - 0.99
	return np.sqrt(delta_low**2 + delta_high**2)
	
start_guess = np.array([0.005,0.001])	

phi_result = scipy.optimize.minimize(func, start_guess, bounds = ((0, None), (0, None)))

phi_result = scipy.optimize.fsolve(func, start_guess, args = (20, 500))