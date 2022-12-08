"""
Units
-----
density (n_e) are in units of cm-3
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import astropy.units as u
import numpy as np

from demcmc.dem import BinnedDEM, TempBins

__all__ = ["EmissionLine"]


@dataclass
class EmissionLine:
    """
    A single emission line.

    Parameters
    ----------
    intensity_obs : float
        Observed intensity.
    sigma_intensity_obs : float
        Uncertainty in observed intensity.
    """

    intensity_obs: Optional[float] = None
    sigma_intensity_obs: Optional[float] = None

    def get_contribution_function_binned(self, temp_bins: TempBins) -> u.Quantity:
        """
        Get contribution function.

        Parameters
        ----------
        temp_bins : TempBins
            Temperature bins to get contribution function at.

        Returns
        -------
        contribution_function : astropy.units.Quantity
            Contribution function at given temperature bins.
        """

    def I_pred(self, dem: BinnedDEM) -> u.Quantity:
        """
        Calculate predicted intensity of a given line.

        Parameters
        ----------
        dem : BinnedDEM
            DEM.

        Returns
        -------
        astropy.units.Quantity
            Predicted intensity.
        """
        cont_func = self.get_contribution_function_binned(dem.temp_bins)
        ret = np.sum(cont_func * dem.values * dem.temp_bins.bin_widths)
        return ret.to_value(u.dimensionless_unscaled)


class GaussianLine(EmissionLine):
    width: u.Quantity[u.K]
    center: u.Quantity[u.K]

    def get_contribution_function_binned(self, temp_bins: TempBins) -> u.Quantity:
        return (
            np.exp(-(((self.center - temp_bins.bin_centers) / self.width) ** 2))
            * u.cm**5
            / u.K
        ) * 1e6


@dataclass
class LineCollection:
    """
    A collection of several emission lines.
    """

    lines: Sequence[EmissionLine]
