"""Automated tests for multimodal weather alignment.

Tests coordinate alignment, timestamp ordering, physical units,
missing-value/gap preservation, and strict anti-leakage causality.
"""
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest

DATASET_PATH = Path("data/processed/weather/aligned_weather_20250201.nc4")
REPORT_PATH = Path("outputs/inspection/weather_alignment_report.json")


@pytest.fixture(scope="module")
def dataset():
    """Fixture providing open handle to aligned weather NetCDF-4 dataset."""
    assert DATASET_PATH.exists(), f"Aligned dataset not found at {DATASET_PATH}"
    with h5py.File(DATASET_PATH, "r") as f:
        yield f


def test_file_exists_and_readable(dataset):
    """Verify that dataset file is openable and contains expected groups/datasets."""
    expected_datasets = [
        "lat", "lon", "time", "reference_time_iso", "nominal_slot_iso",
        "satellite_tir1_mean", "satellite_tir1_min", "satellite_mask",
        "input_precip", "target_precip_lead30", "target_precip_lead60"
    ]
    for dset in expected_datasets:
        assert dset in dataset, f"Missing required dataset: {dset}"


def test_common_grid_coordinates(dataset):
    """Verify common 0.1-degree grid dimensions, bounds, and ordering."""
    lat = dataset["lat"][:]
    lon = dataset["lon"][:]

    assert lat.shape == (25,), f"Expected 25 latitude rows, got {lat.shape}"
    assert lon.shape == (40,), f"Expected 40 longitude columns, got {lon.shape}"

    # Monotonicity
    assert np.all(np.diff(lat) > 0), "Latitude coordinates must be monotonically increasing"
    assert np.all(np.diff(lon) > 0), "Longitude coordinates must be monotonically increasing"

    # Spacing (0.1 degree)
    lat_diffs = np.diff(lat)
    lon_diffs = np.diff(lon)
    assert np.allclose(lat_diffs, 0.1, atol=1e-3), f"Latitude spacing deviates from 0.1°: {lat_diffs}"
    assert np.allclose(lon_diffs, 0.1, atol=1e-3), f"Longitude spacing deviates from 0.1°: {lon_diffs}"

    # Bounding box coverage
    assert lat.min() >= 24.5 and lat.max() <= 27.0, f"Latitude out of Meghalaya bounds: [{lat.min()}, {lat.max()}]"
    assert lon.min() >= 89.5 and lon.max() <= 93.5, f"Longitude out of Meghalaya bounds: [{lon.min()}, {lon.max()}]"


def test_timestamp_ordering_and_cadence(dataset):
    """Verify time steps are strictly chronological with 30-minute nominal intervals."""
    time_epochs = dataset["time"][:]
    assert len(time_epochs) == 9, f"Expected 9 aligned steps, got {len(time_epochs)}"

    # Strictly monotonically increasing
    diffs = np.diff(time_epochs)
    assert np.all(diffs > 0), "Timestamps must be strictly monotonically increasing"

    # Nominal 30-minute cadence (1800s ± 1s jitter for satellite scan end variations)
    for d in diffs:
        assert 1700 <= d <= 1900, f"Interval duration {d}s deviates significantly from 30-min cadence (1800s)"


def test_no_future_information_leakage():
    """Verify zero future information leakage in input rainfall across all aligned steps."""
    assert REPORT_PATH.exists(), f"Report file not found at {REPORT_PATH}"
    import json
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    leakage_audit = report.get("future_leakage_audit", {})
    assert leakage_audit.get("passed") is True, "Future leakage audit failed!"

    for check in leakage_audit.get("step_checks", []):
        assert check["future_leakage_detected"] is False, (
            f"Future leakage detected at slot {check['slot']}: "
            f"input ended at {check['input_rainfall_ended']} after reference time {check['reference_time']}"
        )
        assert check["lead_margin_seconds"] >= 0, (
            f"Negative margin ({check['lead_margin_seconds']}s) at slot {check['slot']}"
        )


def test_observation_gap_preservation(dataset):
    """Verify known INSAT gap slots (02:45, 03:15, 03:45 UTC) are strictly masked without fabrication."""
    sat_mask = dataset["satellite_mask"][:]
    tir1_mean = dataset["satellite_tir1_mean"][:]
    tir1_min = dataset["satellite_tir1_min"][:]

    # Slots 5, 6, 7 correspond to 02:45, 03:15, 03:45 UTC
    gap_indices = [5, 6, 7]
    for idx in gap_indices:
        assert np.all(sat_mask[idx] == 0), f"Step {idx} should have satellite_mask=0"
        assert np.all(np.isnan(tir1_mean[idx])), f"Step {idx} tir1_mean should be all NaN (not fabricated)"
        assert np.all(np.isnan(tir1_min[idx])), f"Step {idx} tir1_min should be all NaN (not fabricated)"


def test_valid_satellite_observations(dataset):
    """Verify that the 6 available INSAT observations are valid, calibrated, and physically sound."""
    sat_mask = dataset["satellite_mask"][:]
    tir1_mean = dataset["satellite_tir1_mean"][:]
    tir1_min = dataset["satellite_tir1_min"][:]

    valid_indices = [0, 1, 2, 3, 4, 8]
    for idx in valid_indices:
        assert np.all(sat_mask[idx] == 1), f"Valid step {idx} should have satellite_mask=1"
        assert np.all(np.isfinite(tir1_mean[idx])), f"Valid step {idx} has unmasked NaNs"

        # Physically valid Kelvin brightness temperatures for tropical/subtropical atmospheric scene
        assert np.nanmin(tir1_mean[idx]) >= 180.0, f"Unphysically low temperature: {np.nanmin(tir1_mean[idx])} K"
        assert np.nanmax(tir1_mean[idx]) <= 330.0, f"Unphysically high temperature: {np.nanmax(tir1_mean[idx])} K"

        # Min cloud top temperature must be <= mean temperature
        assert np.all(tir1_min[idx] <= tir1_mean[idx] + 1e-4), "tir1_min must be <= tir1_mean"


def test_units_and_fill_values(dataset):
    """Verify physical units and missing-value metadata attributes."""
    assert dataset["input_precip"].attrs["units"] == "mm/hr"
    assert dataset["target_precip_lead30"].attrs["units"] == "mm/hr"
    assert dataset["target_precip_lead60"].attrs["units"] == "mm/hr"
    assert dataset["satellite_tir1_mean"].attrs["units"] == "K"
    assert dataset["satellite_tir1_min"].attrs["units"] == "K"

    # Fill values
    assert dataset["input_precip"].attrs["_FillValue"] == -9999.9
    assert np.isnan(dataset["satellite_tir1_mean"].attrs["_FillValue"])

    # Rainfall values must be non-negative (ignoring fill value)
    in_p = dataset["input_precip"][:]
    valid_p = in_p[in_p != -9999.9]
    assert np.all(valid_p >= 0.0), "Precipitation rates must be >= 0.0 mm/hr"


def test_pilot_manifest_integrity():
    """Verify that pilot manifest CSV exists, is well-formed, and confirms anti-leakage margins."""
    import csv
    manifest_path = Path("outputs/inspection/weather_pilot_manifest.csv")
    assert manifest_path.exists(), f"Pilot manifest not found at {manifest_path}"

    with manifest_path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) >= 9, f"Expected at least 9 steps in manifest, got {len(reader)}"
    for row in reader:
        assert "slot" in row
        assert "reference_time_utc" in row
        assert "satellite_status" in row
        assert "anti_leakage_margin_seconds" in row
        assert row["future_leakage_detected"] == "False"
        if row["anti_leakage_margin_seconds"] != "N/A":
            assert int(row["anti_leakage_margin_seconds"]) >= 0, (
                f"Negative margin ({row['anti_leakage_margin_seconds']}s) in row {row}"
            )


def test_pilot_report_integrity():
    """Verify that pilot report JSON contains required metadata, zero-leakage proof, and grid bounds."""
    import json
    report_path = Path("outputs/inspection/weather_pilot_report.json")
    assert report_path.exists(), f"Pilot report not found at {report_path}"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["anti_leakage_causality"]["audit_passed"] is True
    assert report["anti_leakage_causality"]["future_information_leakage"] == "ZERO (VERIFIED)"
    assert report["study_region"]["grid_rows"] == 25
    assert report["study_region"]["grid_cols"] == 40
    assert report["study_region"]["bounding_box_wsen"] == [89.5, 24.5, 93.5, 27.0]
    assert report["observation_counts"]["total_timeline_steps"] >= 9


def test_mosdac_downloader_security_guard():
    """Verify that download_mosdac.py refuses unauthenticated execution and detects missing credentials."""
    import os
    import subprocess
    import sys
    # Copy env but strip any MOSDAC credentials
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("MOSDAC")}
    cmd = [sys.executable, "scripts/download_mosdac.py", "--check-only", "--start-date", "2025-07-01", "--end-date", "2025-07-01"]
    res = subprocess.run(cmd, capture_output=True, text=True, env=clean_env)
    # Must exit with code 2 (credentials missing blocker)
    assert res.returncode == 2, f"Expected exit code 2 (auth blocker), got {res.returncode}: {res.stdout} {res.stderr}"
    assert "EXACT BLOCKER: MOSDAC AUTHENTICATION CREDENTIALS MISSING" in res.stdout


def test_mergir_downloader_discovery():
    """Verify that download_mergir.py queries NASA CMR and discovers all 168 granules for 1-7 July 2025."""
    import sys
    sys.path.insert(0, "scripts")
    import download_mergir
    granules = download_mergir.query_cmr_granules("2025-07-01", "2025-07-07")
    assert len(granules) == 168, f"Expected 168 hourly GPM_MERGIR granules, got {len(granules)}"
    # Verify first and last granule timestamps
    assert granules[0]["time_start"].startswith("2025-07-01")
    assert granules[-1]["time_start"].startswith("2025-07-07")


def test_mergir_downloader_security_guard():
    """Verify that download_mergir.py refuses unauthenticated execution when credentials are missing."""
    import os
    import subprocess
    import sys
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("EARTHDATA")}
    clean_env["EARTHDATA_NO_DOTENV"] = "1"
    cmd = [sys.executable, "scripts/download_mergir.py", "--start-date", "2025-07-01", "--end-date", "2025-07-01"]
    res = subprocess.run(cmd, capture_output=True, text=True, env=clean_env)
    assert res.returncode == 2, f"Expected exit code 2, got {res.returncode}"
    assert "AUTHENTICATION REQUIRED" in res.stdout


def test_mergir_proxy_caveat_and_pipeline_support(tmp_path):
    """Verify that align_weather.py with source=mergir documents the IMERG proxy caveat and grid properties."""
    import subprocess
    import sys
    out_dir = tmp_path / "processed"
    out_insp = tmp_path / "inspection"
    cmd = [
        sys.executable, "scripts/align_weather.py",
        "--source", "mergir",
        "--date", "2025-02-01",
        "--out-dir", str(out_dir),
        "--out-inspection", str(out_insp),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"align_weather failed: {res.stdout} {res.stderr}"
    assert "IR Source: MERGIR" in res.stdout

    # Verify generated dataset contains proxy caveat attribute
    nc_path = out_dir / "aligned_weather_mergir_20250201.nc4"
    assert nc_path.exists(), f"Output file {nc_path} not found"

    with h5py.File(nc_path, "r") as f:
        assert f.attrs["ir_source"] == "MERGIR"
        assert "proxy_caveat" in f.attrs
        assert "proxy-label evaluation" in f.attrs["proxy_caveat"]
        assert f["lat"].shape == (25,)
        assert f["lon"].shape == (40,)


def test_mergir_seven_day_pilot_coverage():
    """Verify that the 7-day MERGIR pilot NetCDF files exist and have consistent dimensions."""
    dates = ["20250701", "20250702", "20250703", "20250704", "20250705", "20250706", "20250707"]
    weather_dir = Path("data/processed/weather")
    for d in dates:
        fpath = weather_dir / f"aligned_weather_mergir_{d}.nc4"
        assert fpath.exists(), f"Missing daily file {fpath}"
        with h5py.File(fpath, "r") as f:
            assert f.attrs["ir_source"] == "MERGIR"
            assert "proxy_caveat" in f.attrs
            assert f["lat"].shape == (25,)
            assert f["lon"].shape == (40,)
            assert f["satellite_tir1_mean"].shape == (48, 25, 40)
            assert f["input_precip"].shape == (48, 25, 40)
            assert f["target_precip_lead30"].shape == (48, 25, 40)
            assert f["target_precip_lead60"].shape == (48, 25, 40)


def test_mergir_seven_day_pilot_manifest_and_anti_leakage():
    """Verify that the 7-day MERGIR pilot manifest has 336 steps and zero future information leakage."""
    import csv
    import json
    manifest_path = Path("outputs/inspection/weather_mergir_pilot_manifest.csv")
    assert manifest_path.exists(), f"Manifest {manifest_path} missing"

    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 336, f"Expected 336 steps in pilot manifest, got {len(rows)}"

    for i, r in enumerate(rows):
        assert r["future_leakage_detected"] == "False", f"Leakage detected at step {i}: {r}"
        if r["anti_leakage_margin_seconds"]:
            margin = float(r["anti_leakage_margin_seconds"])
            assert margin >= 0.0, f"Negative margin at step {i}: {margin}"

    report_path = Path("outputs/inspection/weather_mergir_pilot_report.json")
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["observation_counts"]["total_timeline_steps"] == 336
    assert report["observation_counts"]["satellite_availability_percent"] == 100.0
    assert report["anti_leakage_causality"]["audit_passed"] is True
    assert report["anti_leakage_causality"]["future_information_leakage"] == "ZERO (VERIFIED)"

    preview_path = Path("outputs/inspection/weather_mergir_alignment_preview.png")
    assert preview_path.exists()
    assert preview_path.stat().st_size > 10000


