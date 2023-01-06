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
import xarray as xr
from astropy.visualization import quantity_support

from demcmc.emission import ContFuncDiscrete, EmissionLine

quantity_support()

##################################################################
# Load and plot observed intensities
data_path = Path(os.getcwd()) / "data"
line_intensities = xr.open_dataarray(data_path / "sample_intensity_values.nc")

fig, ax = plt.subplots(constrained_layout=True)
ax.barh(
    line_intensities.coords["Line"],
    line_intensities.loc[:, "Intensity"],
)
ax.set_xlim(0)
ax.set_xlabel("Observed intensity")

##################################################################
# Load and plot calculated contribution functions
cont_funcs = xr.open_dataarray(data_path / "sample_cont_func.nc")

fig, ax = plt.subplots()
for line in cont_funcs.coords["Line"]:
    ax.plot(
        cont_funcs.coords["Temperature"], cont_funcs.loc[line, :], label=line.values
    )
ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("K")
ax.set_ylabel("cm$^{-5}$")
ax.set_title("Contribution functions")


#####
# Create a collection of lines
for line in line_intensities.coords["Line"].values:
    cont_func = cont_funcs.loc[line, :]
    cont_func = ContFuncDiscrete(
        temps=cont_func.coords["Temperature"].values * u.K,
        values=cont_func.values * u.cm**5 / u.K,
    )

    intensity = line_intensities.loc[line, :]
    line = EmissionLine(
        cont_func,
        intensity_obs=intensity.loc["Intensity"].values[0],
        sigma_intensity_obs=intensity.loc["Error"].values[0],
    )

# plt.show()
