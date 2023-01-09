"""
Calculating a DEM from real data
=====================================
This page contains a tutorial, stepping the user through estimating a DEM.
In this tutorial we use a single pixel of real data observed by Hinode/EIS.
"""

##################################################################
# Start by importing the required modules
import os
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from astropy.visualization import quantity_support

from demcmc.emission import EmissionLine, TempBins
from demcmc.io import load_cont_funcs
from demcmc.mcmc import predict_dem_emcee

quantity_support()

##################################################################
# Load intensities and contribution functions
data_path = Path(os.getcwd()) / "data"
line_intensities = xr.open_dataarray(data_path / "sample_intensity_values.nc")
cont_funcs = load_cont_funcs(data_path / "sample_cont_func.nc")

##################################################################
# Plot intensities
fig, ax = plt.subplots(constrained_layout=True)
ax.barh(
    line_intensities.coords["Line"],
    line_intensities.loc[:, "Intensity"],
)
ax.set_xlim(0)
ax.set_xlabel("Observed intensity")

##################################################################
# Plot contribution functions

fig, ax = plt.subplots()
for line in cont_funcs.keys():
    ax.plot(cont_funcs[line].temps, cont_funcs[line].values, label=line)
ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("K")
ax.set_ylabel("cm$^{-5}$")
ax.set_title("Contribution functions")


#####
# Create a collection of lines
lines = []
for line in line_intensities.coords["Line"].values:
    if "Fe" not in line:
        # Only use iron lines
        continue
    cont_func = cont_funcs[line]

    intensity = line_intensities.loc[line, :]
    line = EmissionLine(
        cont_func,
        intensity_obs=intensity.loc["Intensity"].values,
        sigma_intensity_obs=intensity.loc["Error"].values,
    )
    lines.append(line)

temp_bins = TempBins(10 ** np.arange(5.6, 6.8, 0.1) * u.K)
# Run DEM inversion
dem_result = predict_dem_emcee(lines, temp_bins, nsteps=10)

fig, ax = plt.subplots()
# Plot emission loci
for line in lines:
    tbins = TempBins(np.geomspace(1e5, 1e7, 101) * u.K)
    cont_func = line.cont_func.binned(tbins)
    locus = line.intensity_obs / cont_func / tbins.bin_widths
    ax.stairs(locus, tbins.edges, color="k")

# Plot last guess for each walker
for i in range(dem_result.samples.shape[0]):
    ax.stairs(
        dem_result.samples[i, :],
        dem_result.temp_bins.edges,
        color="k",
        alpha=0.1,
        linewidth=1,
    )
ax.set_xscale("log")
ax.set_yscale("log")

fig, ax = plt.subplots()
ax.plot(-dem_result.sampler.get_log_prob())
ax.set_ylabel("-log(probability)")
ax.set_xlabel("Walker step")
ax.set_yscale("log")
plt.show()
