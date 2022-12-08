"""
Units
-----
density (n_e) are in units of cm-3
"""
from dataclasses import dataclass
from typing import Optional

import astropy.units as u
import numpy as np
from plasmapy.particles import Particle

from demcmc.dem import TempBins

__all__ = ["EmissionLine"]


@dataclass
class EmissionLine:
    """
    A single emission line.
    """

    ion: Optional[Particle] = None
    intensity_obs: Optional[float] = None
    sigma_intensity_obs: Optional[float] = None

    @u.quantity_input(T_lower=u.K, T_upper=u.K)
    def get_contribution_function_single(
        self, n_e: u.Quantity, T_lower: u.Quantity, T_upper: u.Quantity
    ) -> u.Quantity:
        """
        Get contribution function, averaged over a given temperature interval.

        Parameters
        ----------
        n_e :
            Electron density.
        T_lower, T_upper :
            Temperature interval bounds.
        """

    def get_contribution_function_binned(self, temp_bins: TempBins) -> u.Quantity:
        """
        Get contribution function across a number of temperature bins.
        """
        return [
            self.get_contribution_function_single(n_e, T_lower, T_upper)
            for T_lower, T_upper in temp_bins.iter_bins()
        ]


class GaussianLine(EmissionLine):
    width: u.Quantity[u.K]
    center: u.Quantity[u.K]

    def get_contribution_function_binned(self, temp_bins: TempBins) -> u.Quantity:
        """
        Get contribution function across a number of temperature bins.
        """
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
