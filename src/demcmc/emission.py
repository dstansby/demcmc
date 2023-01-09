"""
Structures for storing and working with emission lines.
"""
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import astropy.units as u
import numpy as np
import pandas as pd

from demcmc.dem import BinnedDEM, TempBins

__all__ = [
    "ContFunc",
    "ContFuncGaussian",
    "ContFuncDiscrete",
    "EmissionLine",
    "LineCollection",
]


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

    @abstractmethod
    def _binned_arr(self, temp_bins: TempBins) -> np.ndarray:
        """
        Get contribution function averaged over a number of temperature bins.

        Should be returned as a bare array with units of Kelvin.

        Parameters
        ----------
        temp_bins : TempBins
            Temperature bins to get contribution function at.

        Returns
        -------
        astropy.units.Quantity
            Contribution function at given temperature bins.
        """


class ContFuncGaussian(ContFunc):
    """
    A contribution function with a Gaussian profile.

    Parameters
    ----------
    center : u.Quantity
        Center of the Gaussian contribution function.
    width : u.Quantity
        Width of the Gaussian contribution function.
    """

    _units = u.cm**5 / u.K

    def __init__(self, center: u.Quantity, width: u.Quantity):
        self.width = width
        self.center = center

        self._width_MK = self.width.to_value(u.MK)
        self._center_MK = self.center.to_value(u.MK)

    def _binned_arr(self, temp_bins: TempBins) -> np.ndarray:
        """
        Get contribution function without units.
        """
        bins = temp_bins.bin_centers.to_value(u.MK)
        return (
            np.exp(
                -(((self._center_MK - temp_bins._bin_centers_MK) / self._width_MK) ** 2)
            )
        ) * 1e6

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
        return self._binned_arr(temp_bins) * self._units


class ContFuncDiscrete(ContFunc):
    """
    A pre-computed contribution function defined at temperature values.

    Parameters
    ----------
    temps : u.Quantity
        Temperature values of samples.
    values : u.Quantity
        Contribution function values.
    """

    def __init__(self, temps: u.Quantity[u.K], values: u.Quantity[u.cm**5 / u.K]):
        if temps.ndim != 1:
            raise ValueError("temps must be a 1D quantity")
        if values.ndim != 1:
            raise ValueError("values must be a 1D quantity")
        if temps.size != values.size:
            raise ValueError("Temperatures and values must be the same size")

        self._temps = temps
        self._values = values

    @property
    def temps(self) -> u.Quantity[u.K]:
        """
        Temperatures of contribution function samples.
        """
        return self._temps

    @temps.setter
    def temps(self, val: Any) -> None:
        """
        Raises an error.
        """
        raise RuntimeError("ContFuncDiscrete instances are immutable")

    @property
    def values(self) -> u.Quantity[u.cm**5 / u.K]:
        """
        Contribution function values.
        """
        return self._values

    @values.setter
    def values(self, val: Any) -> None:
        """
        Raises an error.
        """
        raise RuntimeError("ContFuncDiscrete instances are immutable")

    def __hash__(self) -> int:
        return id(self)

    def _check_bin_edges(self, temp_bins: TempBins) -> None:
        missing_ts = []
        for t in temp_bins.edges:
            if not np.any(u.isclose(t, self._temps, atol=1 * u.K, rtol=0)):
                missing_ts.append(t)
        if len(missing_ts):
            raise ValueError(
                f"The following bin edges in temp_bins are missing from the contribution function temperature coordinates: {missing_ts}"
            )

    @functools.cache
    def _binned_arr(self, temp_bins: TempBins) -> np.ndarray:
        self._check_bin_edges(temp_bins)

        df = pd.DataFrame(
            {
                "Temps": self.temps.to_value(u.K),
                "values": self.values.to_value(u.cm**5 / u.K),
            }
        )
        df = df.set_index("Temps")
        df["Groups"] = pd.cut(
            df.index, temp_bins.edges.to_value(u.K), include_lowest=True
        )
        means = df.groupby("Groups").mean()["values"].values
        return means

    @functools.cache
    def binned(self, temp_bins: TempBins) -> u.Quantity[u.cm**5 / u.K]:
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
        return self._binned_arr(temp_bins) * u.cm**5 / u.K


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
        cont_func = self.cont_func._binned_arr(dem.temp_bins)
        return np.sum(cont_func * dem._values_arr * dem.temp_bins._bin_widths_arr)


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
