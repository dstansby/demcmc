"""
Structures for storing and working with emission lines.
"""
import functools
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Any, List, Optional

import astropy.units as u
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from demcmc.dem import BinnedDEM, TempBins
from demcmc.units import u_cont_func

__all__ = [
    "ContFunc",
    "ContFuncGaussian",
    "ContFuncDiscrete",
    "EmissionLine",
    "plot_emission_loci",
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

    @abstractproperty
    def temps(self) -> TempBins:
        """
        Default bins at which to evaluate the contribution function when plotting.
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
        return self._binned_arr(temp_bins) * u_cont_func

    @property
    def temps(self) -> TempBins:
        """
        Temperature bins with 100 samples.
        """
        edges = np.linspace(
            self.center - 3 * self.width, self.center + 3 * self.width, 100
        )
        bins: TempBins = TempBins(edges)
        return bins


class ContFuncDiscrete(ContFunc):
    """
    A pre-computed contribution function defined at temperature values.

    Parameters
    ----------
    temps : u.Quantity
        Temperature values of samples.
    values : u.Quantity
        Contribution function values.
    name : str
        Name for the contribution function.
    """

    def __init__(
        self,
        temps: u.Quantity[u.K],
        values: u.Quantity[u.cm**5 / u.K],
        *,
        name: Optional[str] = None,
    ):
        if temps.ndim != 1:
            raise ValueError("temps must be a 1D quantity")
        if values.ndim != 1:
            raise ValueError("values must be a 1D quantity")
        if temps.size != values.size:
            raise ValueError("Temperatures and values must be the same size")

        self._temps = temps
        self._values = values
        self._name = name
        self._hash = id(self)

    def __repr__(self) -> str:
        return f"ContFuncDiscrete(name={self._name}, len(temps)={len(self._temps)})"

    def __str__(self) -> str:
        if self._name is None:
            return f"Discrete contribution function sampled at {len(self._temps)} temperatures"
        else:
            return f"{self._name} contribution function sampled at {len(self._temps)} temperatures"

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
        return self._hash

    def _check_bin_edges(self, temp_bins: TempBins) -> None:
        missing_ts = []
        for t in temp_bins.edges:
            if not np.any(u.isclose(t, self._temps, atol=1 * u.K, rtol=0)):
                missing_ts.append(t)
        if len(missing_ts):
            raise ValueError(
                f"The following bin edges in temp_bins are missing from the contribution function temperature coordinates: {missing_ts}"
            )

    # Can change to just .cache() when Python 3.8 support dropped
    @functools.lru_cache(maxsize=None)
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

    # Can change to just .cache() when Python 3.8 support dropped
    @functools.lru_cache(maxsize=None)
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

    def _I_pred(self, temp_bins: TempBins, dem_values: np.ndarray) -> np.ndarray:
        """
        Same as above, but not using quantities for speed.
        """
        cont_func = self.cont_func._binned_arr(temp_bins)
        return np.sum(cont_func * dem_values * temp_bins._bin_widths_arr)


def plot_emission_loci(lines: List[EmissionLine], ax: Axes, **kwargs: Any) -> None:
    """
    Plot emission loci for a set of observed emission lines.

    Parameters
    ----------
    lines : list[EmissionLine]
        Lines to plot.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    **kwargs : dict
        Keyword arguments are passed to :meth:`~matplotlib.axes.Axes.stairs`.

    Notes
    -----
    Currently only works with lines that have a `ContFuncDiscrete`
    contribution function.
    """
    kwargs.setdefault("color", "k")
    # Plot emission loci
    for line in lines:
        if not isinstance(line.cont_func, ContFuncDiscrete):
            continue
        tbins = TempBins(line.cont_func.temps)
        cont_func = line.cont_func.binned(tbins)
        with np.errstate(over="ignore", divide="ignore"):
            locus = line.intensity_obs / cont_func / tbins.bin_widths
        ax.stairs(locus, tbins.edges, **kwargs)
