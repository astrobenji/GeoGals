'''
George and Emcee

Playground for me to explore how George may be a good fit for our approaches

Created by: Benjamin Metha
Last updated: Mar 24, 2025
'''
# Import relevant packages
# import math
import torch
import gpytorch
#from matplotlib import pyplot as plt
import numpy as np
import pyro
from pyro.infer.mcmc import NUTS, MCMC
import GeoGals as gg
import pandas as pd
import argparse

import emcee  # http://dan.iel.fm/emcee
import corner 

import george as G
from george.modeling import Model

from scipy.optimize import minimize
from scipy.special  import gamma


#######################
## Wrangle real data ##
#######################
gal_ID = 'NGC3351'; diag='O3N2'

Hii_df = gg.open_Hii_df(gal_ID)
meta   = gg.meta_getter(gal_ID)

r_char = 0.4*meta['R25_kpc']
global r_char

wanted_spaxels = ~np.isnan(Hii_df['Z_'+diag]) & ~np.isinf(Hii_df['Z_'+diag])

# Get relevant vectors for Gpytorch
XY  = gg.RA_DEC_to_XY(Hii_df['RA'][wanted_spaxels], Hii_df['DEC'][wanted_spaxels], meta)
Z   = Hii_df['Z_'+diag][wanted_spaxels]
e_Z = Hii_df['e_Z_'+diag][wanted_spaxels]

### Regularise data
Z_mean  = np.mean(Z)
Z_stdev = np.std(Z)

regular_Z   = (Z - Z_mean) / Z_stdev
regular_e_Z = e_Z / Z_stdev


# Convert to numpy arrays
regular_e_Z = np.array(regular_e_Z)
regular_Z   = np.array(regular_Z)
XY = np.array(XY)

# plt.scatter(XY[:,0], XY[:,1], c=Z)


#######################
###### Define GP ######
#######################

# open prior data
prior_data   = gg.read_ICs(gal_ID, diag)
prior_Z_char = prior_data[0]
prior_gradZ  = prior_data[1]

regular_prior_Z_char = (prior_Z_char - Z_mean) / Z_stdev
regular_prior_gradZ  =  prior_gradZ / Z_stdev


# initial vals
phi     = 0.2
log10_a = 0
Z_char  = regular_prior_Z_char
gradZ   = regular_prior_gradZ 

# metric = G.Metric(phi, ndim=2)
a	   = 10**log10_a
kernel = a * G.kernels.ExpKernel(phi, metric_bounds=None, 
			   lower=True, block=None, bounds=None, ndim=2, axes=None)

class MeanModel(Model):
	parameter_names = ("Z_char", 'gradZ')
	
	def get_value(self, XY):
		x_coords = XY[:, 0]	 # First dimension: x
		y_coords = XY[:, 1]	 # Second dimension: y
		r = np.sqrt(x_coords**2 + y_coords**2)	# Compute radial distance
		return self.gradZ * (r.flatten() - r_char) + self.Z_char

gp = G.GP(kernel=kernel, mean=MeanModel(Z_char=Z_char, gradZ=gradZ),
         fit_mean=True)

gp.compute(XY, regular_e_Z)

### Minimize likelihood; solve for parameters

def neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.log_likelihood(Z)
    
# To be TOTALLY CONSISTENT with the tutorial (little changes bb)
def lnprob2(p):
    gp.set_parameter_vector(p)
    return gp.log_likelihood(Z, quiet=True) + gp.log_prior()


### ISSUE in this function: 
### AttributeError: <MeanModelName> object has no attribute 'set_vector' 
### Error in grad log likelihood
### Change to set parameter vector?
###
def grad_neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(Z)
    
p0 = gp.get_parameter_vector()

### Try to set up an emcee run here
ndim, nwalkers = 4, 32
p0 = p0 + 1e-8 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2)

# Run burn-in (80)
p0, lp, _ = sampler.run_mcmc(p0, 80)




    
result = minimize(neg_ln_like, p0, jac=grad_neg_ln_like) # method="L-BFGS-B") if ya wanna get fancy

#Warning: An optimization code something like this should work on most problems 
#but the results can be very sensitive to your choice of initialization and algorithm. 
#If the results are nonsense, try choosing a better initial guess or try a different 
#value of the ``method`` parameter in ``op.minimize``.



# we can apply prior probability functions to the hyperparameters and marginalize
# using Markov chain Monte Carlo (MCMC). To do this, we’ll use the emcee package.

#######################
###### Set priors #####
#######################

# Chosen so that 0.5 dex above/below Z-mean and 0.05 dex/kpc above/below grad-Z
# are unlikely to happen (1% probability each) 
regular_sigma_Z_char = 0.1  / Z_stdev
regular_sigma_gradZ  = 0.02 / Z_stdev

def gamma_prior_20_to_2000(x):
	'''Gamma distribution with 1% probability of being below 0.02 or above 2''' 
	alpha = 1.494
	labda = 2.830
	prob  = np.power(labda, alpha) * np.power(x, alpha - 1) * np.exp(-1.0*labda*x) / gamma(alpha)
	return prob
	
def gamma_prior_tenth_to_ten(x):
	'''Gamma distribution with 1% probability of being below 0.1 or above 10''' 
	alpha = 1.494
	labda = 0.5661
	prob  = np.power(labda, alpha) * np.power(x, alpha - 1) * np.exp(-1.0*labda*x) / gamma(alpha)
	return prob
	
def log_normal_prior(x, mu, sigma):
	return -0.5*( ((x-mu)/sigma)**2 ) - np.log(sigma) - 0.5*np.log(2*np.pi)

def lnprior(Z_char, grad_Z, A, phi):
	# Normal priors on Z_char, grad_Z
	log_Z_char_prior = log_normal_prior(Z_char, mu=regular_prior_Z_char, sigma=regular_sigma_Z_char)
	log_gradZ_prior  = log_normal_prior(grad_Z, mu=regular_prior_gradZ,  sigma=regular_sigma_gradZ)
	# Gamma priors on phi, A
	A_prior = gamma_prior_tenth_to_ten(A)
	phi_prior = gamma_prior_20_to_2000(phi)
	return log_Z_char_prior + log_gradZ_prior + np.log(A_prior) + np.log(phi_prior)

def lnprob(p):
	gp.set_parameter_vector(p)
    lp = lnprior(*p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + gp.lnlikelihood(Z, quiet=True)


def george_lnlike(XY, Z, Z_err, Z_char, log10_a, phi):
	metric = Metric(phi, ndim=2)
	a      = 10**log10_a
	kernel = a * G.kernels.ExpKernel(metric, metric_bounds=None, lower=True, block=None, bounds=None, ndim=2, axes=None)
	gp = G.GP(kernel)
	gp.compute(XY, Z_err)
	resid_Z = Z - gradZ*(R - r_char) - Z_char
	return gp.log_likelihood(resid_Z)
	




########################
##   Read arguments   ##
########################

parser = argparse.ArgumentParser()

parser.add_argument("-g",
		    "--gal-name",
		    help="What is the name of the galaxy you are interested in? Format as Nxxxx.",
            required=True,
            type=str,
            dest='gal_name',
            nargs=1)
            
parser.add_argument("-d",
		    "--diagnostic",
		    help="What SEL diagnostic should be used to estimate the metallicity?",
            required=True,
            type=str,
            dest='diag',
            nargs=1)
            
parser.add_argument("-l",
		    "--chain-length",
		    help="How long will we run these chains for?",
            required=True,
            type=int,
            dest='length',
            nargs=1)
            
args = parser.parse_args()
            
for i in args.gal_name:
	gal_ID = i

for i in args.length:
	N_SAMPLES = i
	
for i in args.diag:
	diag = i

output_path = '../Results/NUTS/testing/'

#######################
## Wrangle real data ##
#######################

Hii_df = gg.open_Hii_df(gal_ID)
meta   = gg.meta_getter(gal_ID)

r_char = 0.4*meta['R25_kpc']
global r_char

wanted_spaxels = ~np.isnan(Hii_df['Z_'+diag]) & ~np.isinf(Hii_df['Z_'+diag])

# Get relevant vectors for Gpytorch
XY  = gg.RA_DEC_to_XY(Hii_df['RA'][wanted_spaxels], Hii_df['DEC'][wanted_spaxels], meta)
Z   = Hii_df['Z_'+diag][wanted_spaxels]
e_Z = Hii_df['e_Z_'+diag][wanted_spaxels]

### Regularise data
Z_mean  = np.mean(Z)
Z_stdev = np.std(Z)

regular_Z   = (Z - Z_mean) / Z_stdev
regular_e_Z = e_Z / Z_stdev

# Convert these to torch tensors
XY    = torch.from_numpy(XY)
Z     = torch.tensor(regular_Z.values)
var_Z = torch.tensor(regular_e_Z.values**2)

# Make them floats, not doubles (else code breaks):
XY    = XY.type(torch.FloatTensor)
Z     = Z.type(torch.FloatTensor)
var_Z = var_Z.type(torch.FloatTensor)

#######################
##	        	     ##
##     Set up GP     ##
##                   ##
#######################

# Create class for mean
class RadialLinearMean(gpytorch.means.Mean):
    def __init__(self):
        super().__init__()
        # Learnable coefficients
        self.gradZ  = torch.nn.Parameter(torch.randn(1))  # Slope
        self.Z_char = torch.nn.Parameter(torch.randn(1))  # Intercept
	
    def forward(self, x):
        # Assuming x has shape [N, D], where D = 2 (at least x and y)
        x_coords = x[:, 0]  # First dimension: x
        y_coords = x[:, 1]  # Second dimension: y
        r = torch.sqrt(x_coords**2 + y_coords**2)  # Compute radial distance
        return self.gradZ * (r - r_char) + self.Z_char

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        # SKI requires a grid size hyperparameter. This util can help with that
        grid_size = gpytorch.utils.grid.choose_grid_size(train_x)

        self.mean_module = RadialLinearMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.MaternKernel(nu=0.5), grid_size=grid_size, num_dims=2
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = var_Z)
model = GPRegressionModel(XY, Z, likelihood)

### ALT: optimize with adam

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Run the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iterations = 100

losses  = []
phis    = []
As      = []
Z_chars = []
gradZs  = []

def train():
	for ii in range(training_iterations):
		optimizer.zero_grad()
		output = model(XY)
		loss = -mll(output, Z)
		losses.append(loss.item())
		loss.backward()
		optimizer.step()
		# Save to running totals
		phis.append(model.covar_module.base_kernel.base_kernel.lengthscale.item())
		As.append(model.covar_module.outputscale.item())
		Z_chars.append(model.mean_module.Z_char.item())
		gradZs.append(model.mean_module.gradZ.item())
		

%time train()

# Access regularised results
phi    = model.covar_module.base_kernel.base_kernel.lengthscale.item()
A      = model.covar_module.outputscale.item()
Z_char = model.mean_module.Z_char.item()
gradZ  = model.mean_module.gradZ.item()

#######################
##	        	     ##
##   Define Priors   ##
##                   ##
#######################

prior_data = gg.read_ICs(gal_ID, diag)
prior_Z_char = prior_data[0]
prior_gradZ  = prior_data[1]

regular_prior_Z_char = (prior_Z_char - Z_mean) / Z_stdev
regular_prior_gradZ  =  prior_gradZ / Z_stdev

# Chosen so that 0.5 dex above/below Z-mean and 0.05 dex/kpc above/below grad-Z
# are unlikely to happen (1% probability each) 
regular_sigma_Z_char = 0.1  / Z_stdev
regular_sigma_gradZ  = 0.02 / Z_stdev

model.mean_module.register_prior("Z_char_prior", gpytorch.priors.NormalPrior(regular_prior_Z_char, regular_sigma_Z_char), "Z_char")
model.mean_module.register_prior("gradZ_prior",  gpytorch.priors.NormalPrior(regular_prior_gradZ,  regular_sigma_gradZ),  "gradZ")

model.covar_module.base_kernel.base_kernel.register_prior("lengthscale_prior", gpytorch.priors.UniformPrior(0.05, 2), "lengthscale")
# Gamma prior values for lengthscale chosen to be 1% chance it's less than 50 pc, or greater than 2000 pc
# model.covar_module.base_kernel.base_kernel.register_prior("lengthscale_prior", gpytorch.priors.GammaPrior(2.095, 3.407), "lengthscale")


# Probably a log-normal prior would be best for outputscale. Try this and see how it goes...
# Try the alternative:
model.covar_module.register_prior("outputscale_prior", gpytorch.priors.UniformPrior(0.1, 10), "outputscale")
# model.covar_module.register_prior("outputscale_prior", gpytorch.priors.LogNormalPrior(0, 2.3), "outputscale")
# model.covar_module.register_prior("outputscale_prior", gpytorch.priors.NormalPrior(1, 0.1), "outputscale")
# Gamma prior values for outputscale chosen to be 1% chance it's less than 0.01, or greater than 100
#model.covar_module.register_prior("outputscale_prior", gpytorch.priors.GammaPrior(1.6, 0.05873), "outputscale")

initial_params = {
'covar_module.base_kernel.base_kernel.lengthscale_prior': torch.tensor(0.2, dtype=torch.float32),
'covar_module.outputscale_prior': torch.tensor(1, dtype=torch.float32),
'mean_module.Z_char_prior': torch.tensor(regular_prior_Z_char, dtype=torch.float32),
'mean_module.gradZ_prior':torch.tensor(regular_prior_gradZ, dtype=torch.float32)
}



#######################
##	        	     ##
##      GO NUTS      ##
##                   ##
#######################

def pyro_model(x, y):
	with gpytorch.settings.fast_computations(False, False, False):
		#with gpytorch.settings.max_cg_iterations(2000):
		sampled_model = model.pyro_sample_from_prior()
		output = sampled_model.likelihood(sampled_model(x))
		pyro.sample("obs", output, obs=y)
	return y

nuts_kernel = NUTS(pyro_model, target_accept_prob=0.9)
mcmc_run = MCMC(nuts_kernel, num_samples=10, initial_params=initial_params,
				disable_progbar=False)
mcmc_run.run(XY, Z)

########################
##	        	      ##
##    Save Results    ##
##                    ##
########################
result_dict = mcmc_run.get_samples()
pd.to_pickle(result_dict, output_path + '{0}_{1}_results.pkl'.format(gal_ID, diag))