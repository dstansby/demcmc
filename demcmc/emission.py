from dataclasses import dataclass

import astropy.units as u
from plasmapy.particles import Particle

from demcmc.dem import TempBins


@dataclass
class EmissionLine:
    """
    A single emission line.
    """

    ion: Particle

    @u.quantity_input
    def get_contribution_function_single(
        self, n_e: u.cm**-3, T_lower: u.K, T_upper: u.K
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

    @u.quantity_input
    def get_contribution_function_binned(
        self, ne: u.cm**-3, temp_bins: TempBins
    ) -> u.Quantity:
        """
        Get contribution function across a number of temperature bins.
        """
