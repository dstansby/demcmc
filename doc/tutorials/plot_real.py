"""
Tutorial
========
This page contains a tutorial, stepping the user through estimating a DEM.
In this tutorial we use a single pixel of real data observed by Hinode/EIS.
"""

######################################################################################
# Start by importing the required modules

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from astropy.visualization import quantity_support

from demcmc.emission import EmissionLine, TempBins, plot_emission_loci
from demcmc.io import load_cont_funcs
from demcmc.mcmc import predict_dem_emcee
from demcmc.sample_data import fetch_sample_data

quantity_support()

######################################################################################
# Load the paths to the sample intensity and contribution function data
intensity_path, cont_func_path = fetch_sample_data()

######################################################################################
# To start with we'll load a set of line intensities. These have been taken from a
# single pixel of a fitted Hinode/EIS intensity map, and saved into a netCDF file for
# easy loading.
line_intensities = xr.open_dataarray(intensity_path)
print(line_intensities)

######################################################################################
# Now we've loaded the intensities, lets do a quick visualisation
fig, ax = plt.subplots(constrained_layout=True)
ax.barh(
    line_intensities.coords["Line"],
    line_intensities.loc[:, "Intensity"],
)
ax.set_xlim(0)
ax.set_xlabel("Observed intensity")

######################################################################################
# As well as observed intensities, the second ingredient we need for estimating a DEM
# is the theoretical contribution functions for each observed line. These have been
# pre-computed and included as sample data. The `demcmc.io.load_cont_funcs` function
# provides functionality to load these from the saved netCDF file.

cont_funcs = load_cont_funcs(cont_func_path)
print(cont_funcs)

######################################################################################
# `cont_funcs` is a dictionary that maps the emission line to a contribution function
# object. This object stores a pre-computed contribution function at a range of
# discrete temperature values. Lets do a visualisation of all the contribution
# functions:
fig, ax = plt.subplots()
for line in cont_funcs:
    ax.plot(cont_funcs[line].temps, cont_funcs[line].values, label=line)

ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
ax.set_title("Contribution functions")

######################################################################################
# Having loaded intensities and contribution functions for each emission line, the
# final preparation step we need to do is create a list `EmissionLine` objects for
# each line. This object stores both the observations and theoretical contribution
# function in a single place for each line.
lines = []
for line in line_intensities.coords["Line"].values:
    if "Fe" not in line:
        # Only use iron lines in the DEM inversion
        continue

    cont_func = cont_funcs[line]
    intensity = line_intensities.loc[line, :]

    line = EmissionLine(
        cont_func,
        intensity_obs=intensity.loc["Intensity"].values,
        sigma_intensity_obs=intensity.loc["Error"].values,
    )
    lines.append(line)

######################################################################################
# Now we can run the DEM inversion! We have to decide what temperature bins we want
# to estimate the DEM in. For this example we'll use 12 bins, with a width of 0.1 MK.
#
# ``demcmc`` uses the `emcee` package to run the MCMC sampler, and attempt to find
# the best values of the DEM that match the line intensity observations.

temp_bins = TempBins(10 ** np.arange(5.6, 6.8, 0.1) * u.K)
# Run DEM inversion
dem_result = predict_dem_emcee(lines, temp_bins, nsteps=10)
print(dem_result)

######################################################################################
# The `DEMResult` class has several useful properties. One of those is ``.samples``,
# which returns the samples of the DEM values at the final step of each MCMC walker.
print(dem_result.samples.shape)
print(dem_result.samples)

######################################################################################
# To visualise the result, we can plot the final samples, and then plot the emission
# loci on the same axes. The emission loci give an upper bound on the DEM.

fig, ax = plt.subplots()

dem_result.plot_final_samples(ax)
plot_emission_loci(lines, ax, color="tab:blue")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(temp_bins.min * 0.9, temp_bins.max * 1.1)
ax.set_ylim(1e18 * u.cm**-5, 1e26 * u.cm**-5)
ax.set_title("Emission loci (blue) and DEM estimates (black)")

plt.show()
