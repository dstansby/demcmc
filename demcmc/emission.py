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

__all__ = ["EmissionLine", "GaussianLine", "LineCollection"]


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
        astropy.units.Quantity
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
    """
    An emission line with a Gaussian contribution function.

    Parameters
    ----------
    center : astropy.units.Quantity
        Center of contribution function.
    width : astropy.units.Quantity
        Width of contribution function.
    """

    center: u.Quantity[u.K]
    width: u.Quantity[u.K]

    _cont_unit = u.cm**5 / u.K

    def __init__(self, center, width):
        self.width = width
        self.center = center

        self._width_MK = self.width.to_value(u.MK)
        self._center_MK = self.center.to_value(u.MK)

    def get_contribution_function_binned(self, temp_bins: TempBins) -> u.Quantity:
        """
        Get contribution function.

        Parameters
        ----------
        temp_bins : TempBins
            Temperature bins to get contribution function at.

        Returns
        -------
        astropy.units.Quantity
            Contribution function at given temperature bins.
        """
        bins = temp_bins.bin_centers.to_value(u.MK)
        return (
            (
                np.exp(
                    -(
                        ((self._center_MK - temp_bins._bin_centers_MK) / self._width_MK)
                        ** 2
                    )
                )
            )
            * 1e6
            * self._cont_unit
        )


@dataclass
class LineCollection:
    """
    A collection of several emission lines.

    Parameters
    ----------
    lines : Sequence[EmissionLine]
        Emission lines.
    """

    lines: Sequence[EmissionLine]
