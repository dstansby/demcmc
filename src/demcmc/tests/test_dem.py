import astropy.units as u
import numpy as np
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

    def test_setters(self, temp_bins: TempBins) -> None:
        with pytest.raises(RuntimeError):
            temp_bins.edges = [2, 3] * u.K

    def test_centers(self, temp_bins: TempBins) -> None:
        assert np.all(temp_bins.bin_centers == [1.5, 3] * u.K)

    def test_iter_bins(self, temp_bins: TempBins) -> None:
        expected = [(1, 2) * u.K, (2, 4) * u.K]
        for b, e in zip(temp_bins.iter_bins(), expected):
            assert np.all(b == e)
