import astropy.units as u
import numpy as np
import pytest
import xarray as xr

from demcmc.dem import DEMOutput
from demcmc.emission import EmissionLine, TempBins
from demcmc.io import load_cont_funcs
from demcmc.mcmc import predict_dem_emcee
from demcmc.sample_data import fetch_sample_data


@pytest.fixture
def cont_funcs():
    _, cont_func_path = fetch_sample_data()
    return load_cont_funcs(cont_func_path)


@pytest.fixture
def line_intensities():
    line_intensities_path, _ = fetch_sample_data()
    return xr.load_dataarray(line_intensities_path)


@pytest.fixture
def lines(line_intensities, cont_funcs):
    line_intensities, cont_funcs

    lines = []
    for line in line_intensities.coords["Line"].values:
        cont_func = cont_funcs[line]
        intensity = line_intensities.loc[line, :]

        line = EmissionLine(
            cont_func,
            intensity_obs=intensity.loc["Intensity"].values,
            sigma_intensity_obs=intensity.loc["Error"].values,
        )
        lines.append(line)

    return lines


def test_mcmc(lines, tmpdir):
    """
    Smoke test of the MCMC run. Does NOT check values are correct or sensible.
    """
    temp_bins = TempBins(10 ** np.arange(5.6, 6.8, 0.1) * u.K)
    dem_result = predict_dem_emcee(lines, temp_bins, nsteps=1)
    assert isinstance(dem_result, DEMOutput)

    save_path = tmpdir / "result.nc"
    dem_result.save(save_path)
    assert save_path.exists()

    DEMOutput.load(save_path)
