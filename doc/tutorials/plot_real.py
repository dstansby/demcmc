"""
Calculating a DEM from real data
=====================================
This page contains a tutorial, stepping the user through estimating a DEM.
In this tutorial we use a single pixel of real data observed by Hinode/EIS.
"""
from pathlib import Path

##################################################################
# Start by importing the required modules
import matplotlib.pyplot as plt
import xarray as xr
from astropy.visualization import quantity_support


quantity_support()

##################################################################
# Load and plot observed intensities
line_intensities = xr.open_dataarray(
    Path(__file__).parent / "data" / "sample_intensity_values.nc"
)

fig, ax = plt.subplots(constrained_layout=True)
ax.scatter(
    line_intensities.loc[:, "Intensity"], line_intensities.coords["Line"], marker="x"
)
ax.set_xlim(0)
ax.set_xlabel("Observed intensity")
ax.yaxis.grid()

plt.show()
