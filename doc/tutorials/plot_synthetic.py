"""
Calculating a DEM from synthetic data
=====================================
This page contains a tutorial, stepping the user through estimating a DEM.
We don't use any real data here, but instead a series of fake lines that have Gaussian contribution functions.
"""
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import quantity_support

from demcmc.dem import BinnedDEM, TempBins
from demcmc.emission import GaussianLine

quantity_support()

##################################################################
# To start with we'll create a set of syntemtic emission lines.
# When these are created we set the center and width of their
# contribution functions.
lines = []
# 20 lines, evenly spaced from 1 MK -> 2 MK
line_centers = np.linspace(1, 2, 21) * u.MK
line_width = 0.1 * u.MK

for line_center in line_centers:
    line = GaussianLine()
    line.center = line_center
    line.width = line_width
    lines.append(line)

temp_bins = TempBins(np.linspace(0.5, 2.5, 601) * u.MK)

fig, ax = plt.subplots()
for line in lines:
    ax.stairs(line.get_contribution_function_binned(temp_bins), temp_bins.edges)

ax.set_title("Line contribution functions")

##################################################################
# Now lets create a 'fake' input DEM. We will use this to simulate line
# intensities that each of these emission lines would observe.
#
# This input DEM is calculated on a coarse temperature grid. This is
# because it's easier to infer fewer DEM values using the MCMC inversion
# later.
coarse_temps = TempBins(np.linspace(1, 2, 11) * u.MK)
dem_in = np.exp(-(((coarse_temps.bin_centers - 1.2 * u.MK) / (0.2 * u.MK)) ** 2))
dem_in = BinnedDEM(coarse_temps, dem_in * u.cm**-5)

fig, axs = plt.subplots(nrows=2, sharex=True)
ax = axs[0]
ax.stairs(dem_in.values, coarse_temps.edges)
ax.set_title("Input DEM")

ax = axs[1]
for line in lines:
    ax.stairs(line.get_contribution_function_binned(temp_bins), temp_bins.edges)

ax.set_title("Line contribution functions")

plt.show()
