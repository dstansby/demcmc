"""
Functions for carrying out MCMC estimation of DEMs.
"""
from multiprocessing import Pool

import astropy.units as u
import emcee
import numpy as np

from demcmc.dem import BinnedDEM, TempBins
from demcmc.emission import EmissionLine

__all__ = []


def I_pred(line: EmissionLine, dem: BinnedDEM) -> u.Quantity:
    """
    Calculate predicted intensity of a given line.
    """
    cont_func = line.get_contribution_function_binned(dem.temp_bins)
    ret = np.sum(cont_func * dem.values * dem.temp_bins.bin_widths)
    return ret.to_value(u.dimensionless_unscaled)


def _log_prob_line(
    line: EmissionLine,
    dem: BinnedDEM,
) -> float:
    """
    Get log probability of intensity stored in ``line`` for the given DEM.
    """
    intensity_pred = I_pred(line, dem)
    # print(line.intensity_obs, intensity_pred)
    ret = -float(
        ((line.intensity_obs - intensity_pred) / line.sigma_intensity_obs) ** 2
    )
    return ret


def _log_prob_lines(
    lines: list[EmissionLine],
    dem: BinnedDEM,
) -> float:
    """
    Get log probability of all line intensities stored in ``lines`` for the given DEM.
    """
    probbs = [_log_prob_line(line, dem) for line in lines]
    # print(probbs)
    # print(np.sum(probbs))
    return np.sum(probbs)


def log_prob(dem_vals, temp_bins, lines):
    """
    log probability of a given set of (log10(DEM values)).
    The DEM values are passed as logs to enforce positivity.
    """
    if np.any(dem_vals < 0):
        return -np.inf
    dem = BinnedDEM(temp_bins, dem_vals * u.cm**-5)
    p = _log_prob_lines(lines, dem)
    return p


def predict_dem(
    lines: list[EmissionLine],
    temp_bins: TempBins,
    nsteps=10,
):
    """
    Given a list of emission lines (which each have contribution functions
    and observed intensities), estimate the true DEM in the bins given by
    temp_bins.
    """
    ndim = len(temp_bins)
    # Set number of bin walkers to twice dimensionality of the parameter space
    nwalkers = 2 * ndim + 1

    dem_guess = 0.5 + 0.1 * np.random.rand(nwalkers, ndim)
    # Create sampler
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob, args=[temp_bins, lines], pool=pool
        )
        # Run sampler
        sampler.run_mcmc(dem_guess, nsteps, progress=True)

    return sampler
