"""
Functions for carrying out MCMC estimation of DEMs.
"""
from typing import List, Sequence, Tuple

import emcee
import numpy as np

from demcmc.dem import DEMOutput, TempBins
from demcmc.emission import EmissionLine

__all__ = ["predict_dem_emcee"]


def _log_prob_line(
    line: EmissionLine,
    temp_bins: TempBins,
    dem_guess: np.ndarray,
) -> float:
    """
    Get log probability of intensity stored in ``line`` for the given DEM.

    Returns
    -------
    float
        Probability.
    """
    intensity_pred = line._I_pred(temp_bins, dem_guess)
    ret = -float(
        ((line.intensity_obs - intensity_pred) / line.sigma_intensity_obs) ** 2
    )
    return ret


def _log_prob_lines(
    lines: List[EmissionLine],
    temp_bins: TempBins,
    dem_guess: np.ndarray,
) -> float:
    """
    Get log probability of all line intensities stored in ``lines`` for the given DEM.

    Returns
    -------
    float
        Probability.
    """
    probbs = [_log_prob_line(line, temp_bins, dem_guess) for line in lines]
    return float(np.sum(probbs))


def _log_prob_single_variation(
    dem_val: float,
    idx_varied: int,
    dem_guess: np.ndarray,
    temp_bins: TempBins,
    lines: List[EmissionLine],
) -> float:
    """
    log probability of a given set of DEM values, varying one of them.
    The DEM values are passed as logs to enforce positivity.]

    Parameter
    ---------
    dem_val :
        DEM value being varied.
    idx_varied :
        Index of dem_guess that is being varied.
    dem_guess :
        Rest of the DEM values.

    Returns
    -------
    float
        Probability.
    """
    dem_guess[idx_varied] = dem_val
    return _log_prob(dem_guess, temp_bins, lines)


def _log_prob(
    dem_guess: np.ndarray,
    temp_bins: TempBins,
    lines: List[EmissionLine],
) -> float:
    """
    log probability of a given set of DEM values, varying one of them.
    The DEM values are passed as logs to enforce positivity.]

    Parameter
    ---------
    dem_val :
        DEM value being varied.
    dem_guess :
        Rest of the DEM values.

    Returns
    -------
    float
        Probability.
    """
    if np.any(dem_guess < 0):
        return float(-np.inf)

    p = _log_prob_lines(lines, temp_bins, dem_guess)
    return p


def predict_dem_emcee(
    lines: Sequence[EmissionLine],
    temp_bins: TempBins,
    *,
    nsteps: int,
    nwalkers: int,
    progress: bool = True,
) -> DEMOutput:
    """
    Estimate DEM from a number of emission lines.

    Parameters
    ----------
    lines : Sequence[EmissionLine]
        Emission lines.
    temp_bins : TempBins
        Temperature bins to predict DEM in.
    nsteps : int
        Total number of steps for the MCMC walkers to take.
    nwalkers : int
        Number of MCMC walkers to use.
    progress : bool
        Whether to show a progress bar for the MCMC walking.

    Returns
    -------
    DEMOutput
        Output container.
    """
    # Initial DEM value guesses
    n_dem = len(temp_bins)
    dem_guess = 1e22 * np.ones(n_dem)

    # Start by running emcee on each of the parameters individually
    #
    # This speeds up getting started because there instead of searching
    # an N-dimensional space, the search is done on N 1-dimensional
    # spaces.
    dem_guess, _ = _vary_values_independently(lines, temp_bins, dem_guess, nsteps=100)

    # Now run MCMC across the ful N-dimensional space to get the final guess
    dem_guess = np.repeat(np.atleast_2d(dem_guess), nwalkers, axis=0)
    dem_guess += np.random.rand(*dem_guess.shape) * 0.01 * dem_guess

    sampler = emcee.EnsembleSampler(nwalkers, n_dem, _log_prob, args=[temp_bins, lines])
    sampler.run_mcmc(dem_guess, nsteps, progress=progress)

    return DEMOutput._from_sampler(sampler, temp_bins)


def _vary_values_independently(
    lines: Sequence[EmissionLine],
    temp_bins: TempBins,
    dem_guess: np.ndarray,
    *,
    nsteps: int,
) -> Tuple[np.ndarray, List[emcee.EnsembleSampler]]:
    ndim = 1
    nwalkers = 3
    n_dem = len(temp_bins)

    parameter_guess = np.repeat(np.atleast_2d(dem_guess), nwalkers, axis=0)
    # Add randomness to initial guesses
    parameter_guess += np.random.rand(*parameter_guess.shape) * 0.01 * parameter_guess

    samplers = []
    for i in range(n_dem):
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            _log_prob_single_variation,
            args=[i, dem_guess, temp_bins, lines],
        )
        # Run sampler
        sampler.run_mcmc(parameter_guess[:, i : i + 1], nsteps=nsteps, progress=False)
        samplers.append(sampler)

        samples = sampler.get_chain()
        # Take mean of the last 10 steps
        dem_guess[i] = np.mean(samples[-10:, :, :])
        parameter_guess[:, i] = samples[-1, :, 0]

    return dem_guess, samplers
