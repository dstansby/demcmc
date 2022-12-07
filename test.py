import astropy.units as u
import numpy as np
from plasmapy.particles import Particle

from demcmc.dem import BinnedDEM, TempBins
from demcmc.emission import EmissionLine
from demcmc.mcmc import _log_prob_line


if __name__ == '__main__':
    temps = TempBins(np.logspace(5, 8, 31) * u.K)
    line = EmissionLine(Particle("Fe XII"))
    ne = 1e8 * u.cm**-3
    dem = BinnedDEM(temps, 30 * np.ones(len(temps)))

    intensity_obs = 1
    sigma_obs = 0.1

    p = _log_prob_line(line, ne, dem, intensity_obs, sigma_obs)
    print(p)
