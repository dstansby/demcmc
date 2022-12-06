"""
Structures for working with DEM data.
"""
from dataclasses import dataclass

import astropy.units as u
import numpy as np

__all__ = ["BinnedDEM", "TempBins"]


@dataclass
class TempBins:
    """
    A set of temperature bins.
    """

    bin_edges: u.Quantity

    @property
    def bin_widths(self) -> u.Quantity:
        """
        Widths of the bins.
        """
        return np.diff(self.bin_edges)

    def __len__(self):
        """
        Number of bins.
        """
        return self.bin_edges.size - 1


@dataclass
class BinnedDEM:
    """
    A DEM binned over a range of temperature values.

    The binning is equal in log-space.
    """

    temp_bins: TempBins
    values: u.Quantity
