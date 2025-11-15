'''
KISSGP_test.py

Script for learning KISS-GP with a real data example.
Uses GPytorch's KISS GP method to solve for the hyperparameters
Testing on the data file for NGC 3351 with O3N2 as the diagnostic (our smallest galaxy)
Uses adam as the optimizer -- but will be extended to NUTS in a future script

Created by: Benjamin Metha
Last updated: Mar 04, 2025
'''
# Import relevant packages
# import math
import torch
import gpytorch
#from matplotlib import pyplot as plt
import numpy as np
import GeoGals as gg

# Create this class (it's important)
class RadialLinearMean(gpytorch.means.Mean):
    def __init__(self):
        super().__init__()
        # Learnable coefficients; initialise to reasonable guesses
        # to speed up computation
        self.a = -0.03 + 0.03*torch.nn.Parameter(torch.randn(1))  # Slope
        self.b = 8.7 + torch.nn.Parameter(torch.randn(1))  # Intercept

    def forward(self, x):
        # Assuming x has shape [N, D], where D = 2 (at least x and y)
        x_coords = x[:, 0]  # First dimension: x
        y_coords = x[:, 1]  # Second dimension: y
        r = torch.sqrt(x_coords**2 + y_coords**2)  # Compute radial distance
        return self.a * r + self.b


# Wrangle real data
gal_ID = 'NGC3351'
diag   = 'O3N2'

Hii_df = gg.open_Hii_df(gal_ID)
meta   = gg.meta_getter(gal_ID)

wanted_spaxels = ~np.isnan(Hii_df['Z_'+diag]) & ~np.isinf(Hii_df['Z_'+diag])

# Get relevant vectors for Gpytorch
XY  = gg.RA_DEC_to_XY(Hii_df['RA'][wanted_spaxels], Hii_df['DEC'][wanted_spaxels], meta)
Z   = Hii_df['Z_'+diag][wanted_spaxels]
e_Z = Hii_df['e_Z_'+diag][wanted_spaxels]

# Convert these to torch tensors
XY    = torch.from_numpy(XY)
Z     = torch.tensor(Z.values)
var_Z = torch.tensor(e_Z.values**2)

# Set up GP
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
#likelihood = gpytorch.likelihoods.GaussianLikelihood() # a more basic option
model = GPRegressionModel(XY, Z, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Run the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iterations = 10

def train():
    for ii in range(training_iterations):
        optimizer.zero_grad()
        output = model(XY)
        loss = -mll(output, Z)
        loss.backward()
        optimizer.step()


# Access results
phi   = model.covar_module.base_kernel.base_kernel.lengthscale.item()
A     = model.covar_module.outputscale.item()
Z_c   = model.mean_module.b.item()
gradZ = model.mean_module.a.item()

print('After optimization ({0} loops):'.format(training_iterations))
print('Z_c = {0}'.format(Z_c))
print('gradZ = {0}'.format(gradZ))
print('phi = {0}'.format(phi))
print('A = {0}'.format(A))
