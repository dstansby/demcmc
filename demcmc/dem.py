"""
Structures for working with DEM data.
"""
from dataclasses import dataclass

import astropy.units as u


__all__ = ['BinnedDEM', 'TempBins']


@dataclass
class TempBins:
    """
    A set of temperature bins.
    """

    @property
    def bin_widths(self) -> u.Quantity:
        """
        Widths of the bins.
        """


@dataclass
class BinnedDEM:
    """
    A DEM binned over a range of temperature values.

    The binning is equal in log-space.
    """

    temp_bins: TempBins
    values: u.Quantity
