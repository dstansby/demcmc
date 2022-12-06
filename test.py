import astropy.units as u
import numpy as np
from plasmapy.particles import Particle

from demcmc.dem import BinnedDEM, TempBins
from demcmc.emission import EmissionLine
from demcmc.mcmc import _log_prob_line

temps = TempBins(np.logspace(5, 8, 31) * u.K)
line = EmissionLine(Particle("Fe XII"))
ne = 1e8 * u.cm**-3
print(len(temps))
dem = BinnedDEM(temps, 30 * np.ones(len(temps)))
print(dem)

intensity_obs = 1
sigma_obs = 0.1

p = _log_prob_line(line, ne, dem, intensity_obs, sigma_obs)
