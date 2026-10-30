"""Automated tests for Dynamic Landslide Hazard Index (NASA LHASA 2.0)."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

DYNAMIC_NC = Path("data/processed/landslide/meghalaya_dynamic_hazard_july2025.nc4")


def test_dynamic_hazard_netcdf_structure():
    """Verify that dynamic hazard NetCDF exists, has required datasets and metadata."""
    assert DYNAMIC_NC.exists(), f"Missing {DYNAMIC_NC}"

    with h5py.File(DYNAMIC_NC, "r") as f:
        # Attributes
        assert "NASA LHASA 2.0" in f.attrs["framework"]
        assert "Research prototype" in f.attrs["disclaimer"]
        assert f.attrs["peak_hazard_step"] == 316

        # Required datasets
        expected = [
            "dynamic_hazard_series_weather_grid",
            "antecedent_trigger_series_mm",
            "reference_time_iso",
            "lat_weather_grid",
            "lon_weather_grid",
            "lsi_weather_grid",
            "peak_dynamic_hazard_90m",
            "peak_antecedent_rainfall_90m",
            "lat_90m",
            "lon_90m",
            "meghalaya_mask_90m",
        ]
        for key in expected:
            assert key in f, f"Missing dataset: {key}"


def test_dynamic_hazard_values_and_categorical_levels():
    """Verify that hazard levels are valid categorical levels {0, 1, 2, 3}."""
    with h5py.File(DYNAMIC_NC, "r") as f:
        haz_series = f["dynamic_hazard_series_weather_grid"][:]
        a_series = f["antecedent_trigger_series_mm"][:]
        haz_90m = f["peak_dynamic_hazard_90m"][:]
        mask_90m = f["meghalaya_mask_90m"][:]

    assert haz_series.shape == (336, 25, 40)
    assert a_series.shape == (336, 25, 40)

    # All levels must be in {0, 1, 2, 3}
    assert np.all(np.isin(haz_series, [0, 1, 2, 3]))
    assert np.all(np.isin(haz_90m, [0, 1, 2, 3]))

    # Antecedent rain must be non-negative
    assert np.all(a_series >= 0.0)

    # In 90m peak state, masked pixels outside Meghalaya must be 0
    assert np.all(haz_90m[mask_90m == 0] == 0)


def test_dynamic_hazard_coordinate_orientations():
    """Verify coordinate orientation alignment between weather and 90m grids."""
    with h5py.File(DYNAMIC_NC, "r") as f:
        lat_w = f["lat_weather_grid"][:]
        lon_w = f["lon_weather_grid"][:]
        lat_90 = f["lat_90m"][:]
        lon_90 = f["lon_90m"][:]

    # Weather grid coordinates are ascending
    assert lat_w[1] > lat_w[0]
    assert lon_w[1] > lon_w[0]

    # 90m grid latitude is descending (standard North to South raster), lon is ascending
    assert lat_90[1] < lat_90[0]
    assert lon_90[1] > lon_90[0]

    # Geolocation bounds cover Meghalaya
    assert lat_w[0] <= 24.6 and lat_w[-1] >= 26.9
    assert lon_w[0] <= 89.6 and lon_w[-1] >= 93.4
