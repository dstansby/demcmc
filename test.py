import astropy.units as u
import numpy as np
from plasmapy.particles import Particle

from demcmc.dem import BinnedDEM, TempBins
from demcmc.emission import EmissionLine
from demcmc.mcmc import _log_prob_line

if __name__ == "__main__":
    temps = TempBins(np.logspace(5, 8, 31) * u.K)
    line = EmissionLine(Particle("Fe XII"))
    ne = 1e8 * u.cm**-3
    dem = BinnedDEM(temps, 30 * np.ones(len(temps)) * u.cm**-5)

    intensity_obs = 7e6
    sigma_obs = intensity_obs / 10

    def get_contribution_function_binned(n_e: u.cm**-3, temp_bins: TempBins):
        """
        Create a dummy gaussian contribution function.
        """
        center = 1 * u.MK
        width = 0.1 * u.MK

        # WARNING: Not sure if units are right here but setting them to make the code work
        return (
            np.exp(-((center - temp_bins.edges[:-1]) ** 2) / (width**2))
            * u.cm**5
            / u.K
        )

    line.get_contribution_function_binned = get_contribution_function_binned

    p = _log_prob_line(line, ne, dem, intensity_obs, sigma_obs)
    print(p)
