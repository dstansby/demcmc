"""
Units
-----
density (n_e) are in units of cm-3
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import astropy.units as u
import numpy as np
from plasmapy.particles import Particle

from demcmc.dem import TempBins

__all__ = ["EmissionLine"]


@dataclass
class EmissionLine(ABC):
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

    @abstractmethod
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
