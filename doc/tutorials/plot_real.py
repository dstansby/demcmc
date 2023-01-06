"""
Calculating a DEM from real data
=====================================
This page contains a tutorial, stepping the user through estimating a DEM.
In this tutorial we use a single pixel of real data observed by Hinode/EIS.
"""
import os
from pathlib import Path

##################################################################
# Start by importing the required modules
import matplotlib.pyplot as plt
import xarray as xr
from astropy.visualization import quantity_support

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

"""
for line_center in line_centers:
    line = GaussianLine(line_center, line_width)
    lines.append(line)"""
plt.show()
