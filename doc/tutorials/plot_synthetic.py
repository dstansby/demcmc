"""
Calculating a DEM from synthetic data
=====================================
This page contains a tutorial, stepping the user through estimating a DEM.
We don't use any real data here, but instead a series of fake lines that have Gaussian contribution functions.
"""
##################################################################
# Start by importing the required modules
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import quantity_support

from demcmc.dem import BinnedDEM, TempBins
from demcmc.emission import ContFuncGaussian, EmissionLine
from demcmc.mcmc import predict_dem_emcee

quantity_support()

##################################################################
# To start with we'll create a set of syntemtic emission lines.
# These are created with their line center and line width, which
# define their Gaussian contribution functions.
lines = []

# 20 lines, evenly spaced from 1 MK -> 2 MK
line_centers = np.linspace(1, 2, 21) * u.MK
line_width = 0.1 * u.MK

for line_center in line_centers:
    cont_func = ContFuncGaussian(line_center, line_width)
    lines.append(EmissionLine(cont_func))

# 600 temperature bins, linearly spaced from 0.5 -> 2.5 MK
temp_bins = TempBins(np.linspace(0.5, 2.5, 601) * u.MK)

fig, ax = plt.subplots()
for line in lines:
    ax.stairs(
        line.cont_func.get_contribution_function_binned(temp_bins), temp_bins.edges
    )

ax.set_title("Line contribution functions")

##################################################################
# Now lets create a 'fake' input DEM. We will use this to simulate
# the intensities that each of these lines would emit.
#
# This input DEM is calculated on a very coarse temperature grid,
# mainly to make the DEM inversion later much faster for this
# example.
coarse_temps = TempBins(np.linspace(1, 2, 6) * u.MK)
dem_in = np.exp(-(((coarse_temps.bin_centers - 1.2 * u.MK) / (0.2 * u.MK)) ** 2))
dem_in = BinnedDEM(coarse_temps, dem_in * u.cm**-5)

fig, axs = plt.subplots(nrows=2, sharex=True)
ax = axs[0]
ax.stairs(dem_in.values, coarse_temps.edges)
ax.set_title("Input DEM")

ax = axs[1]
for line in lines:
    ax.stairs(
        line.cont_func.get_contribution_function_binned(temp_bins), temp_bins.edges
    )

ax.set_ylim(bottom=0)
ax.set_title("Line contribution functions")

##################################################################
# Now lets use this DEM and the line contribution functions to
# simulate the intensity that each line would observe.
for line in lines:
    I_pred = line.I_pred(dem_in)
    line.intensity_obs = I_pred
    # Set error to 1/10th of observation
    line.sigma_intensity_obs = I_pred / 10


centers = u.Quantity([line.cont_func.center for line in lines])
intensities = u.Quantity([line.intensity_obs for line in lines])

fig, ax = plt.subplots(constrained_layout=True)
ax.scatter(centers, intensities)
ax.set_title("Simulated line intensities")
ax.set_xlabel("Line contribution function center / MK")

##################################################################
# Now pretend you didn't see the input DEM above! The problem we
# want to solve is:
#
#   Given the line contribution functions, and the observed
#   intensity in each line, what was the original DEM?
#
# To do this we use the ``predict_dem_emcee`` function to
# do a DEM inversion. This takes the list of lines we created
# earlier, and the temperature bins that we want to estimate
# the DEM in.
if __name__ == "__main__":
    # Run DEM inversion
    sampler = predict_dem_emcee(lines, dem_in.temp_bins, nsteps=1000)
    # Get results
    samples = sampler.get_chain()

    fig, ax = plt.subplots()
    # Plot last guess for each walker
    for i in range(samples.shape[1]):
        ax.stairs(
            samples[-1, i, :], dem_in.temp_bins.edges, color="k", alpha=0.1, linewidth=1
        )
    ax.scatter(dem_in.temp_bins.bin_centers, dem_in.values, label="Input DEM")
    ax.set_yscale("log")
    ax.legend()

    fig, ax = plt.subplots()
    ax.plot(-sampler.get_log_prob())
    ax.set_ylabel("-log(probability)")
    ax.set_xlabel("Walker step")
    ax.set_yscale("log")
    plt.show()
