"""Unit and integration tests for Step 8: Landslide Terrain & Antecedent Rainfall Ingestion."""
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.append(os.path.abspath("scripts"))
from download_dem import MEGHALAYA_TILES, verify_dem_tile


def test_downloaded_dem_tiles_integrity():
    """Verify that all 8 NASA SRTM 30 m NetCDF tiles exist and pass HDF5 schema validation."""
    dem_dir = Path("data/raw/dem")
    assert dem_dir.exists(), "DEM directory data/raw/dem does not exist"

    for tile in MEGHALAYA_TILES:
        fpath = dem_dir / f"{tile}.SRTMGL1_NC.nc"
        assert fpath.exists(), f"Missing required DEM tile: {fpath}"
        assert verify_dem_tile(fpath), f"DEM tile failed integrity check: {fpath}"


def test_boundary_geojson_integrity():
    """Verify authoritative geoBoundaries Meghalaya GeoJSON validity and bounds."""
    boundary_path = Path("data/raw/boundary/meghalaya_boundary.geojson")
    assert boundary_path.exists(), "Missing boundary GeoJSON"

    with open(boundary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("type") == "FeatureCollection"
    assert len(data.get("features", [])) >= 1
    feat = data["features"][0]
    geom = feat.get("geometry", {})
    assert geom.get("type") in ["Polygon", "MultiPolygon"]

    # Verify bounding coordinates encompass Meghalaya
    all_pts = []
    def extract_pts(c):
        if isinstance(c[0], (int, float)):
            all_pts.append(c)
        else:
            for sub in c:
                extract_pts(sub)
    extract_pts(geom["coordinates"])

    lons = [pt[0] for pt in all_pts]
    lats = [pt[1] for pt in all_pts]

    assert 89.5 <= min(lons) <= 90.0
    assert 92.5 <= max(lons) <= 93.0
    assert 24.9 <= min(lats) <= 25.2
    assert 26.0 <= max(lats) <= 26.3


def test_terrain_90m_netcdf_features():
    """Verify processed 90 m terrain NetCDF datasets, units, and physical value ranges."""
    nc_path = Path("data/processed/terrain/meghalaya_terrain_features_90m.nc4")
    assert nc_path.exists(), f"Missing 90 m terrain NetCDF: {nc_path}"
    assert nc_path.stat().st_size > 10_000_000

    required_keys = [
        "lat", "lon",
        "elevation_mean", "elevation_min", "elevation_max", "relief",
        "slope_mean", "slope_max", "slope_std",
        "aspect_northness", "aspect_eastness",
        "curvature_mean", "meghalaya_mask", "valid_mask",
    ]

    with h5py.File(nc_path, "r") as f:
        for k in required_keys:
            assert k in f, f"Missing dataset {k} in {nc_path}"

        elev = f["elevation_mean"][:]
        slope = f["slope_mean"][:]
        northness = f["aspect_northness"][:]
        eastness = f["aspect_eastness"][:]
        m_mask = f["meghalaya_mask"][:]

        # Masked checks inside Meghalaya
        megh_elev = elev[m_mask == 1]
        assert np.min(megh_elev) >= -50.0
        assert np.max(megh_elev) <= 2500.0

        megh_slope = slope[m_mask == 1]
        assert np.min(megh_slope) >= 0.0
        assert np.max(megh_slope) <= 90.0

        # Northness and Eastness in [-1, 1]
        assert np.all((northness >= -1.01) & (northness <= 1.01))
        assert np.all((eastness >= -1.01) & (eastness <= 1.01))


def test_terrain_weather_grid_alignment():
    """Verify alignment of coarse terrain summary onto (25, 40) 0.1-degree weather grid."""
    nc_path = Path("data/processed/terrain/meghalaya_terrain_aligned_weather_grid.nc4")
    assert nc_path.exists()

    with h5py.File(nc_path, "r") as f:
        assert f["lat"].shape == (25,)
        assert f["lon"].shape == (40,)
        assert f["elevation_mean"].shape == (25, 40)
        assert f["slope_mean"].shape == (25, 40)
        assert f["terrain_ruggedness_index"].shape == (25, 40)
        assert f["meghalaya_coverage"].shape == (25, 40)

        # Coverage must be in [0, 1]
        cov = f["meghalaya_coverage"][:]
        assert np.all((cov >= 0.0) & (cov <= 1.0))
        assert np.any(cov > 0.0)  # Must cover Meghalaya


def test_antecedent_rainfall_netcdf():
    """Verify antecedent rainfall accumulation NetCDF features and accumulations."""
    nc_path = Path("data/processed/landslide/antecedent_rainfall_july2025.nc4")
    assert nc_path.exists()

    with h5py.File(nc_path, "r") as f:
        assert f["lat"].shape == (25,)
        assert f["lon"].shape == (40,)
        assert len(f["reference_time_iso"]) == 336

        for k in ["accum_6h", "accum_12h", "accum_24h", "accum_72h", "accum_7d"]:
            assert k in f
            ds = f[k][:]
            assert ds.shape == (336, 25, 40)
            # Verify accumulation is non-negative where finite
            valid_vals = ds[np.isfinite(ds)]
            assert np.all(valid_vals >= 0.0)

        # Peak 7-day accumulation must be substantial (> 100 mm in monsoon)
        max_7d = np.nanmax(f["accum_7d"][:])
        assert max_7d >= 100.0


def test_landslide_inventory_audit_integrity():
    """Verify that the landslide inventory contains verified events and reports scientific caveats."""
    csv_path = Path("outputs/landslide/meghalaya_landslide_events.csv")
    json_path = Path("outputs/landslide/landslide_inventory_audit.json")

    assert csv_path.exists()
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["total_meghalaya_events"] >= 50
    assert audit["reliably_dated_events"] >= 50
    assert "scientific_assessment" in audit
    assert audit["scientific_assessment"]["can_train_24h_forecasting"] is False
    assert "susceptibility" in audit["scientific_assessment"]["recommendation"].lower()
