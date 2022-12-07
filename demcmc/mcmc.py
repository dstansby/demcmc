"""
Functions for carrying out MCMC estimation of DEMs.
"""
import astropy.units as u
import numpy as np

from demcmc.dem import BinnedDEM
from demcmc.emission import EmissionLine

__all__ = []


@u.quantity_input(n_e=u.cm**-3)
def _I_pred(line: EmissionLine, n_e: u.Quantity, dem: BinnedDEM) -> u.Quantity:
    """
    Calculate predicted intensity of a given line.
    """
    cont_func = line.get_contribution_function_binned(n_e, dem.temp_bins)
    return np.sum(cont_func * dem.values * dem.temp_bins.bin_widths)


@u.quantity_input(n_e=u.cm**-3)
def _log_prob_line(
    line: EmissionLine,
    n_e: u.Quantity,
    dem: BinnedDEM,
) -> float:
    """
    Get log probability given line intensity.
    """
    intensity_pred = _I_pred(line, n_e, dem)
    return -float(
        ((line.intensity_obs - intensity_pred) / line.sigma_intensity_obs) ** 2
    )
