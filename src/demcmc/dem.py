"""
Classes for working with DEM data.
"""
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Iterator, Tuple

import astropy.units as u
import emcee
import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from demcmc.units import u_dem, u_temp

__all__ = ["BinnedDEM", "TempBins", "DEMOutput"]


class TempBins:
    """
    A set of temperature bins.

    The bins are defined through the bin edges.

    Parameters
    ----------
    edges : astropy.units.Quantity
        Bin edges.
    """

    @u.quantity_input(edges=u_temp)
    def __init__(self, edges: u.Quantity):
        self._edges = edges
        self._edges_arr = edges.to_value(u_temp)

    @property
    def edges(self) -> u.Quantity[u_temp]:
        """
        Edges of the temperature bins.
        """
        return self._edges

    @edges.setter
    def edges(self, val: Any) -> None:
        """
        Raises an error.
        """
        raise RuntimeError("ContFuncDiscrete instances are immutable")

    @property
    def min(self) -> u.Quantity[u.K]:
        """
        Lower bound of the temperature bins.
        """
        return self.edges.min()

    @property
    def max(self) -> u.Quantity[u.K]:
        """
        Upper bound of the temperature bins.
        """
        return self.edges.max()

    def __hash__(self) -> int:
        return id(self)

    @cached_property
    def bin_widths(self) -> u.Quantity:
        """
        Widths of the bins.
        """
        return np.diff(self.edges)

    @cached_property
    def _bin_widths_arr(self) -> np.ndarray:
        """
        Widths of the bins as a bare array.
        """
        return np.diff(self._edges_arr)

    @cached_property
    def bin_centers(self) -> u.Quantity:
        """
        Centers of the bins.
        """
        return (self.edges[:-1] + self.edges[1:]) / 2

    @cached_property
    def _bin_centers_MK(self) -> np.ndarray:
        """
        Centers of the bins as bare numbers in units of MK.
        """
        return self.bin_centers.to_value(u.MK)

    def __len__(self) -> int:
        """
        Number of bins.
        """
        # int is just to make mypy happy
        return int(self.edges.size - 1)

    def iter_bins(self) -> Iterator[Tuple[u.Quantity, u.Quantity]]:
        """
        Iterate through lower/upper bounds of temperature bins.

        Yields
        ------
        lower_edge : astropy.units.Quantity
            Lower edge of bin.
        upper_edge : astropy.units.Quantity
            Upper edge of bin.
        """
        for i in range(len(self)):
            yield self.edges[i], self.edges[i + 1]


@dataclass
class BinnedDEM:
    """
    A DEM binned over a range of temperature values.

    Parameters
    ----------
    temp_bins : TempBins
        Temperature bins.
    values : astropy.units.Quantity
        DEM values.
    """

    temp_bins: TempBins
    values: u.Quantity

    @u.quantity_input(values=u.cm**-5)
    def __init__(self, temp_bins: TempBins, values: u.Quantity):
        self.temp_bins = temp_bins
        self.values = values

        self._values_arr = self.values.to_value(u_dem)


@dataclass
class DEMOutput:
    """
    Output from running DEM calculation.

    This is not intended to be created by users.
    """

    def __init__(self, sampler: emcee.EnsembleSampler, temp_bins: TempBins) -> None:
        self._sampler = sampler
        self._temp_bins = temp_bins

    @property
    def sampler(self) -> emcee.EnsembleSampler:
        return self._sampler

    @property
    def temp_bins(self) -> TempBins:
        """
        Temperature bins.
        """
        return self._temp_bins

    @property
    def samples(self) -> u.Quantity:
        """
        Return the last set of samples from the walker.
        """
        return self.sampler.get_chain()[-1, :, :] * u_dem

    def plot_final_samples(self, ax: Axes, **kwargs: Any) -> None:
        """
        Plot the final samples of the MCMC walker.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes to plot the samples on.
        kwargs :
            Any keyword arguments are passed to `~matplotlib.axes.Axes.stairs`.
        """
        kwargs.setdefault("color", "k")
        kwargs.setdefault("alpha", 0.1)
        kwargs.setdefault("linewidth", 1)
        for i in range(self.samples.shape[0]):
            ax.stairs(self.samples[i, :], self.temp_bins.edges, **kwargs)

    def save(self, path: Path) -> None:
        """
        Save a computed DEM to a netCDF file.

        Parameters
        ----------
        """
        temp_centers = self.temp_bins.bin_centers
        temp_edges = self.temp_bins.edges

        samplers = np.arange(self.samples.shape[0])
        coords = {"Sampler": samplers, "Temp bin center": temp_centers.to_value(u_temp)}

        da = xr.DataArray(
            data=self.samples,
            coords=coords,
            attrs={"Temp bin edges": temp_edges.to_value(u_temp)},
        )
