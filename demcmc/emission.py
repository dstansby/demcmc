from dataclasses import dataclass

import astropy.units as u
from plasmapy.particles import Particle

from demcmc.dem import TempBins

__all__ = ["EmissionLine"]


@dataclass
class EmissionLine:
    """
    A single emission line.
    """

    ion: Particle

    @u.quantity_input(ne=u.cm**-3, T_lower=u.K, T_upper=u.K)
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

    @u.quantity_input(n_e=u.cm**-3)
    def get_contribution_function_binned(
        self, n_e: u.Quantity, temp_bins: TempBins
    ) -> u.Quantity:
        """
        Get contribution function across a number of temperature bins.
        """
        return [
            self.get_contribution_function_single(n_e, T_lower, T_upper)
            for T_lower, T_upper in temp_bins.iter_bins()
        ]
