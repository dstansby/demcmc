"""
Functions for carrying out MCMC estimation of DEMs.

Units
-----
density (n_e) are in units of cm-3
"""
import astropy.units as u
import numpy as np

from demcmc.dem import BinnedDEM
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
    intensity_pred = _I_pred(line, dem)
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
