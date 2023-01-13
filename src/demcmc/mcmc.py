"""
Functions for carrying out MCMC estimation of DEMs.
"""
from typing import Sequence

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
    lines: list[EmissionLine],
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
    lines: list[EmissionLine],
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
    lines: list[EmissionLine],
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
    nsteps: int = 10,
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
        Number of steps for each MCMC walker to take. This is the number
        of steps initial parameter guessing takes. The multi-dimensional
        walker then takes ``nsteps * len(temp_bins)`` steps in the final
        part.

    Returns
    -------
    DEMOutput
        Output container.

    Notes
    -----
    - The number of walkers is automatically set to twice the number of
      temperature bins plus one.
    - The initial guess for the DEM is uniform across temperature bins.
    """
    n_dem = len(temp_bins)
    nwalkers = 2 * n_dem + 1
    # Initial DEM value guesses
    dem_guess = 1e22 * np.random.rand(nwalkers, n_dem)

    # Start by running emcee on each of the parameters individually
    #
    # This speeds up getting started because there instead of searching
    # an N-dimensional space, the search is done on N 1-dimensional
    # spaces.
    for i in range(n_dem):
        ndim = 1
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            _log_prob_single_variation,
            args=[i, np.mean(dem_guess, axis=0), temp_bins, lines],
        )
        # Run sampler
        param_guess = dem_guess[:, i].reshape((nwalkers, 1))
        sampler.run_mcmc(param_guess, nsteps=100, progress=progress)

        samples = sampler.get_chain()
        # Take average of last two steps across all samplers
        dem_guess[:, i] = samples[-1, :, 0]

    # Now run MCMC across the ful N-dimensional space to get the final guess
    sampler = emcee.EnsembleSampler(nwalkers, n_dem, _log_prob, args=[temp_bins, lines])
    sampler.run_mcmc(dem_guess, nsteps * n_dem, progress=progress)
    return DEMOutput._from_sampler(sampler, temp_bins)
