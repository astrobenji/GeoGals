'''
KISSGP_test.py

Script for learning KISS-GP with a mock data example.
Uses GPytorch's KISS GP method to solve for the hyperparameters

Uses a radial kernel with missing points and a matern function
Uses FixedNoiseGaussianLikelihood for heteroskedastic error

Created by: Benjamin Metha
Last updated: Mar 11, 2025 (input NUTS)
'''

import torch
import gpytorch
import math
import pyro
from pyro.infer.mcmc import NUTS, MCMC

########################
## Make training data ##
########################

n = 40
EPS = 1e-3

# We make an nxn grid of training points spaced every 1/(n-1) on [0,1]x[0,1]
train_x = torch.zeros(pow(n, 2), 2)
for i in range(n):
    for j in range(n):
        train_x[i * n + j][0] = float(i) / (n-1)
        train_x[i * n + j][1] = float(j) / (n-1)
        
# Change train_x to run between -5 and 5 (why not)
train_x = 10*train_x - 5

# Make training data (y = 8.8 - 0.1x + sine stuff + random error)
train_y = torch.sin((train_x[:, 0] + train_x[:, 1]) * (2 * math.pi)) + torch.randn_like(train_x[:, 0]).mul(0.01)
train_r = torch.sqrt(train_x[:, 0]**2 + train_x[:,1]**2)
train_y = train_y + 8.8 - 0.1*train_r

# Random variance of each Y point
var_y = torch.randn(train_y.shape)**2 + EPS

# Take only 500 pixels, selected at random:
sample_indices = torch.randperm(1600)[:500] 

train_x = train_x[sample_indices]
train_y = train_y[sample_indices]
var_y   = var_y[sample_indices]

#####################
## Setup GP models ##
#####################

# Create this class for a radially varying mean
class RadialLinearMean(gpytorch.means.Mean):
    def __init__(self):
        super().__init__()
        # Learnable coefficients
        self.a = torch.nn.Parameter(torch.randn(1))  # Slope
        self.b = torch.nn.Parameter(torch.randn(1))  # Intercept

    def forward(self, x):
        # Assuming x has shape [N, D], where D = 2 (at least x and y)
        x_coords = x[:, 0]  # First dimension: x
        y_coords = x[:, 1]  # Second dimension: y
        
        r = torch.sqrt(x_coords**2 + y_coords**2)  # Compute radial distance
        return self.a * r + self.b
        
# Setup GP regression model, with
#  * radially varying mean
#  * SKI powered exponential kernel
#  * fixed heteroskedastic noise
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
        
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = var_y)
model = GPRegressionModel(train_x, train_y, likelihood)

#######################
##	        	     ##
##   Define Priors   ##
##                   ##
#######################

model.mean_module.register_prior("intercept_prior", gpytorch.priors.NormalPrior(8.7, 0.25), "b")
model.mean_module.register_prior("slope_prior",  gpytorch.priors.NormalPrior(0.0, 0.2),  "a")

model.covar_module.base_kernel.base_kernel.register_prior("lengthscale_prior", gpytorch.priors.GammaPrior(2.095, 0.003407), "lengthscale")
model.covar_module.register_prior("outputscale_prior", gpytorch.priors.GammaPrior(1.6, 0.05873), "outputscale")

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

nuts_kernel = NUTS(pyro_model)
mcmc_run = MCMC(nuts_kernel, num_samples=10, warmup_steps=1, disable_progbar=False)
mcmc_run.run(train_x, train_y)

####################
## Access results ##
####################

phi   = model.covar_module.base_kernel.base_kernel.lengthscale.item()
A     = model.covar_module.outputscale.item()
Z_c   = model.mean_module.b.item()
gradZ = model.mean_module.a.item()

print('After optimization ({0} loops):'.format(training_iterations))
print('Z_c = {0}'.format(Z_c))
print('gradZ = {0}'.format(gradZ))
print('phi = {0}'.format(phi))
print('A = {0}'.format(A))