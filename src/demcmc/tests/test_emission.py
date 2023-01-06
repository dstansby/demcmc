import astropy.units as u
import pytest

from demcmc.dem import TempBins
from demcmc.emission import ContFuncDiscrete


@pytest.fixture
def cont_func() -> ContFuncDiscrete:
    temps = [1, 2, 3, 4, 5] * u.MK
    values = [0.1, 0.3, 0.5, 0.7, 0.5] * u.cm**5 / u.K
    return ContFuncDiscrete(temps, values)


class TestContFuncDiscrete:
    def test_binned_error(self, cont_func):
        tbins = TempBins([0, 1, 3, 4] * u.MK)
        with pytest.raises(
            ValueError,
            match="The following bin edges in temp_bins are missing from the contribution function temperature coordinates",
        ):
            cont_func.binned(tbins)

    def test_binned(self, cont_func):
        tbins = TempBins([1, 3, 5] * u.MK)
        assert u.allclose(cont_func.binned(tbins), [0.3, 0.6] * u.cm**5 / u.K)
