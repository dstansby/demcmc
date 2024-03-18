"""
Load sample data.
"""

from pathlib import Path
from typing import Tuple

import pooch

REGISTRY = pooch.create(
    path=pooch.os_cache("demcmc"),
    base_url="doi:10.5281/zenodo.7534284/",
    registry={
        "sample_cont_func.nc": "md5:6bf547459d83d24b0da2736f36380afb",
        "sample_intensity_values.nc": "md5:f8c9481705cd0295cf1ad0f4fa41caf5",
    },
)


def fetch_sample_data() -> Tuple[Path, Path]:
    return REGISTRY.fetch("sample_intensity_values.nc"), REGISTRY.fetch(
        "sample_cont_func.nc"
    )
