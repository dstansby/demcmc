"""
Structures for working with DEM data.
"""
from dataclasses import dataclass
from typing import Iterator, Tuple

import astropy.units as u
import numpy as np

__all__ = ["BinnedDEM", "TempBins"]


@dataclass
class TempBins:
    """
    A set of temperature bins.
    """

    edges: u.Quantity

    @u.quantity_input(edges=u.K)
    def __init__(self, edges: u.Quantity):
        self.edges = edges

    @property
    def bin_widths(self) -> u.Quantity:
        """
        Widths of the bins.
        """
        return np.diff(self.edges)

    def __len__(self) -> int:
        """
        Number of bins.
        """
        # int is just to make mypy happy
        return int(self.edges.size - 1)

    def iter_bins(self) -> Iterator[Tuple[u.Quantity, u.Quantity]]:
        """
        Iterate through lower/upper bounds of temperature bins.
        """
        for i in range(len(self)):
            yield self.edges[i], self.edges[i + 1]


@dataclass
class BinnedDEM:
    """
    A DEM binned over a range of temperature values.

    The binning is equal in log-space.
    """

    temp_bins: TempBins
    values: u.Quantity

    @u.quantity_input(values=u.cm**-5)
    def __init__(self, temp_bins: TempBins, values: u.Quantity):
        self.temp_bins = temp_bins
        self.values = values
