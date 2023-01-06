import astropy.units as u
import pytest

from demcmc.dem import TempBins


@pytest.fixture
def temp_bins() -> TempBins:
    return TempBins([1, 2, 4] * u.K)


class TestTempBins:
    def test_widths(self, temp_bins: TempBins) -> None:
        assert u.allclose(temp_bins.bin_widths, [1, 2] * u.K)

    def test_len(self, temp_bins: TempBins) -> None:
        assert len(temp_bins) == 2

    def test_wrong_units(self) -> None:
        with pytest.raises(u.UnitsError):
            TempBins([1, 2] * u.cm)
