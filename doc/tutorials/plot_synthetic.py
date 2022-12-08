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

from demcmc.dem import TempBins
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
plt.show()
