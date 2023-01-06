"""
Structures for storing and working with emission lines.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import astropy.units as u
import numpy as np

from demcmc.dem import BinnedDEM, TempBins

__all__ = ["EmissionLine", "GaussianLine", "LineCollection"]


class ContFunc(ABC):
    """
    A contribution function.
    """

    @abstractmethod
    def binned(self, temp_bins: TempBins) -> u.Quantity:
        """
        Get contribution function averaged over a number of temperature bins.

        Parameters
        ----------
        temp_bins : TempBins
            Temperature bins to get contribution function at.

        Returns
        -------
        astropy.units.Quantity
            Contribution function at given temperature bins.
        """


class ContFuncGaussian:
    """
    A contribution function with a Gaussian profile.
    """

    def __init__(self, center: u.Quantity, width: u.Quantity):
        self.width = width
        self.center = center

        self._width_MK = self.width.to_value(u.MK)
        self._center_MK = self.center.to_value(u.MK)

    def binned(self, temp_bins: TempBins) -> u.Quantity:
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
            * u.cm**5
            / u.K
        )


class ContFuncDiscrete(ContFunc):
    """
    A pre-computed contribution function defined at a number of discrete
    temperature values.
    """


@dataclass
class EmissionLine:
    """
    A single emission line.

    Parameters
    ----------
    cont_func : ContFunc
        Contribution function.
    intensity_obs : float
        Observed intensity.
    sigma_intensity_obs : float
        Uncertainty in observed intensity.
    """

    cont_func: ContFunc
    intensity_obs: Optional[float] = None
    sigma_intensity_obs: Optional[float] = None

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
        cont_func = self.cont_func.binned(dem.temp_bins)
        ret = np.sum(cont_func * dem.values * dem.temp_bins.bin_widths)
        return ret.to_value(u.dimensionless_unscaled)


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
