from demcmc.dem import TempBins
import pytest
import astropy.units as u

@pytest.fixture
def temp_bins():
    return TempBins([1, 2, 4] * u.K)

class TestTempBins:
    def test_widths(self, temp_bins):
        assert u.allclose(temp_bins.bin_widths, [1, 2] * u.K)

    def test_len(self, temp_bins):
        assert len(temp_bins) == 2
