"""
Functions for carrying out MCMC estimation of DEMs.
"""
from typing import Sequence

import astropy.units as u
import emcee
import numpy as np

from demcmc.dem import BinnedDEM, TempBins
from demcmc.emission import EmissionLine

__all__ = ["predict_dem"]


def _log_prob_line(
    line: EmissionLine,
    dem: BinnedDEM,
) -> float:
    """
    Get log probability of intensity stored in ``line`` for the given DEM.
    """
    intensity_pred = line.I_pred(dem)
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
    return float(np.sum(probbs))


def _log_prob(
    dem_vals: np.ndarray, temp_bins: TempBins, lines: list[EmissionLine]
) -> float:
    """
    log probability of a given set of (log10(DEM values)).
    The DEM values are passed as logs to enforce positivity.
    """
    if np.any(dem_vals < 0):
        return float(-np.inf)
    dem = BinnedDEM(temp_bins, dem_vals * u.cm**-5)
    p = _log_prob_lines(lines, dem)
    return p


def predict_dem(
    lines: Sequence[EmissionLine],
    temp_bins: TempBins,
    nsteps: int = 10,
) -> emcee.EnsembleSampler:
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
    # with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_prob, args=[temp_bins, lines])
    # Run sampler
    sampler.run_mcmc(dem_guess, nsteps, progress=True)

    return sampler
