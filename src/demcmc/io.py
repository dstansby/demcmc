from pathlib import Path

import astropy.units as u
import xarray as xr

from demcmc.emission import ContFuncDiscrete

__all__ = ["load_cont_funcs"]


def load_cont_funcs(path: Path) -> dict[str, ContFuncDiscrete]:
    """
    Load a set of contribution functions from a netCDF file.

    The file should contain a 2D array of data, with coordinates named
    "Temperature" for the temperatures and "Line" for the line names.

    Parameters
    ----------
    path : os.PathLike
        Path of the netCDF file.

    Returns
    -------
    dict[str, ContFuncDiscrete]
        Mapping of line name to the loaded contribution function.
    """
    da = xr.open_dataarray(path)
    temps = da.coords["Temperature"].values
    lines = da.coords["Line"].values

    cont_funcs = {}
    for line in lines:
        cont_funcs[line] = ContFuncDiscrete(
            temps=temps * u.K,
            values=da.loc[line, :].values * u.cm**5 / u.K,
        )

    return cont_funcs
