import astropy.units as u
import emcee
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import quantity_support
from plasmapy.particles import Particle

quantity_support()

from demcmc.dem import BinnedDEM, TempBins
from demcmc.emission import GaussianLine
from demcmc.mcmc import _log_prob_lines

if __name__ == "__main__":
    # Create temperature bins
    temps = TempBins(np.logspace(5, 7, 21) * u.K)

    # Create emission line
    intensity_obs = 7e5
    sigma_obs = intensity_obs / 10

    lines = []
    for line_center in np.logspace(-1, 1, 41) * u.MK:
        line = GaussianLine(Particle("Fe XII"), intensity_obs, sigma_obs)
        line.center = line_center
        line.width = 0.2 * u.MK
        lines.append(line)

    ne = 1e8 * u.cm**-3
    dem_guess = np.ones(len(temps)) + np.random.rand(len(temps)) * 0.01
    dem = BinnedDEM(temps, dem_guess * u.cm**-5)

    # Plot contribution function
    fig, ax = plt.subplots()
    for line in lines:
        ax.stairs(line.get_contribution_function_binned(ne, temps), temps.edges)
    ax.set_yscale("log")

    def log_prob(dem_vals):
        # Enforce positivity
        if np.any(dem_vals < 0):
            return -np.inf
        dem.values = dem_vals * u.cm**-5
        p = _log_prob_lines(lines, ne, dem)
        print(p)
        return p

    nwalkers = 40
    shape = (nwalkers, len(temps))
    initial_vals = np.ones(shape) + np.random.rand(*shape) * 0.1
    ndim = len(dem.values)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(initial_vals, 5)

    samples = sampler.get_chain(flat=True)

    fig, ax = plt.subplots()
    for i in range(samples.shape[1]):
        ax.plot(samples[:, i])
    ax.set_ylabel("Parameter value")
    ax.set_xlabel("MCMC step")

    plt.show()
