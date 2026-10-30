"""Landslide Feature Engineering & Terrain Processing Pipeline.

Processes official NASA SRTM 30 m DEM tiles, masks with the authoritative Meghalaya state
boundary, computes scientifically documented terrain morphometry (slope, aspect, curvature,
roughness), aggregates to intermediate (90 m) and weather-coupled (0.1 deg) modeling grids,
and extracts multi-timescale antecedent rainfall accumulation features from IMERG.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


# Authoritative Study Area Bounding Box with buffer
AOI_BOUNDS = {
    "min_lat": 25.00,
    "max_lat": 26.15,
    "min_lon": 89.75,
    "max_lon": 92.85,
}

R_EARTH = 6371000.0  # Earth mean radius in meters


def build_mosaic_from_tiles(dem_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seamlessly assemble the 8 SRTMGL1 NetCDF tiles into a 2x4 regional mosaic.
    
    Rows:
      Row 0 (North): Lat 26-27, Lon 89-93 (N26E089, N26E090, N26E091, N26E092)
      Row 1 (South): Lat 25-26, Lon 89-93 (N25E089, N25E090, N25E091, N25E092)
    Returns:
      (mosaic_array, lats, lons) where shape is (7201, 14401)
    """
    tile_names = [
        ["N26E089", "N26E090", "N26E091", "N26E092"],
        ["N25E089", "N25E090", "N25E091", "N25E092"],
    ]

    # Full mosaic dimension: 2 degrees lat (7201 pixels), 4 degrees lon (14401 pixels)
    mosaic = np.zeros((7201, 14401), dtype=np.int16)

    for r_idx in range(2):
        for c_idx in range(4):
            t_name = tile_names[r_idx][c_idx]
            fpath = dem_dir / f"{t_name}.SRTMGL1_NC.nc"
            if not fpath.exists():
                raise FileNotFoundError(f"Missing required DEM tile: {fpath}")

            with h5py.File(fpath, "r") as hf:
                tile_dem = hf["SRTMGL1_DEM"][:]  # (3601, 3601)

            # Row bounds in mosaic (row 0 is north, row 1 is south)
            row_start = r_idx * 3600
            row_end = row_start + 3601
            col_start = c_idx * 3600
            col_end = col_start + 3601

            mosaic[row_start:row_end, col_start:col_end] = tile_dem

    lats = np.linspace(27.0, 25.0, 7201, dtype=np.float64)
    lons = np.linspace(89.0, 93.0, 14401, dtype=np.float64)
    return mosaic, lats, lons


def crop_to_aoi(
    mosaic: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    bounds: Dict[str, float] = AOI_BOUNDS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop stitched regional mosaic to the Meghalaya AOI bounding box."""
    # Latitudes are descending
    lat_mask = (lats <= bounds["max_lat"]) & (lats >= bounds["min_lat"])
    # Longitudes are ascending
    lon_mask = (lons >= bounds["min_lon"]) & (lons <= bounds["max_lon"])

    row_indices = np.where(lat_mask)[0]
    col_indices = np.where(lon_mask)[0]

    r_min, r_max = row_indices[0], row_indices[-1] + 1
    c_min, c_max = col_indices[0], col_indices[-1] + 1

    cropped_dem = mosaic[r_min:r_max, c_min:c_max]
    cropped_lats = lats[r_min:r_max]
    cropped_lons = lons[c_min:c_max]

    return cropped_dem, cropped_lats, cropped_lons


def rasterize_boundary_mask(
    boundary_geojson_path: Path,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Rasterize official Meghalaya polygon onto target lat/lon grid using scanline fill."""
    with open(boundary_geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    feature = data["features"][0]
    geom = feature["geometry"]
    # Handle Polygon or MultiPolygon
    if geom["type"] == "Polygon":
        rings = [geom["coordinates"][0]]
    elif geom["type"] == "MultiPolygon":
        rings = [poly[0] for poly in geom["coordinates"]]
    else:
        raise ValueError(f"Unsupported geometry type: {geom['type']}")

    H = len(lats)
    W = len(lons)
    min_lon, max_lon = lons[0], lons[-1]
    max_lat, min_lat = lats[0], lats[-1]  # descending

    img = Image.new("1", (W, H), 0)
    draw = ImageDraw.Draw(img)

    for ring in rings:
        px_coords = []
        for pt in ring:
            c = (pt[0] - min_lon) / (max_lon - min_lon) * (W - 1)
            r = (max_lat - pt[1]) / (max_lat - min_lat) * (H - 1)
            px_coords.append((c, r))
        draw.polygon(px_coords, fill=1)

    mask = np.array(img, dtype=np.uint8)
    return mask


def compute_terrain_morphometry(
    dem: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute slope, aspect, northness, eastness, and curvature using Horn's finite differences."""
    H, W = dem.shape
    z = dem.astype(np.float32)

    # Valid elevation mask (SRTM void fill is -32768)
    valid_mask = ((dem >= -50) & (dem <= 8000)).astype(np.uint8)

    # Metric pixel spacings in meters
    dlat_deg = np.abs(lats[1] - lats[0])
    dlon_deg = np.abs(lons[1] - lons[0])
    dy = R_EARTH * (dlat_deg * np.pi / 180.0)

    lats_rad = lats * (np.pi / 180.0)
    dx = R_EARTH * np.cos(lats_rad)[:, None] * (dlon_deg * np.pi / 180.0)  # (H, 1)

    # Allocate derivative arrays
    dz_dx = np.zeros_like(z)
    dz_dy = np.zeros_like(z)
    d2z_dx2 = np.zeros_like(z)
    d2z_dy2 = np.zeros_like(z)

    # Horn (1981) 3x3 finite differences (interior pixels)
    # x-direction (East-West)
    dz_dx[1:-1, 1:-1] = (
        (z[:-2, 2:] + 2.0 * z[1:-1, 2:] + z[2:, 2:])
        - (z[:-2, :-2] + 2.0 * z[1:-1, :-2] + z[2:, :-2])
    ) / (8.0 * dx[1:-1])

    # y-direction (North-South; row 0 is North, so row :-2 is North, 2: is South)
    dz_dy[1:-1, 1:-1] = (
        (z[:-2, :-2] + 2.0 * z[:-2, 1:-1] + z[:-2, 2:])
        - (z[2:, :-2] + 2.0 * z[2:, 1:-1] + z[2:, 2:])
    ) / (8.0 * dy)

    # Slope in degrees
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.rad2deg(slope_rad)

    # Aspect in degrees (0 = North, 90 = East, 180 = South, 270 = West)
    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    aspect_deg = (np.rad2deg(aspect_rad) + 360.0) % 360.0

    # Flat terrain (< 0.5 degree slope) has undefined aspect -> set to 0
    flat_mask = slope_deg < 0.5
    aspect_deg[flat_mask] = 0.0

    # Cyclic aspect encoding (Northness and Eastness)
    aspect_northness = np.cos(aspect_rad)
    aspect_eastness = np.sin(aspect_rad)
    aspect_northness[flat_mask] = 0.0
    aspect_eastness[flat_mask] = 0.0

    # Total Laplacian Curvature (scaled by 100)
    d2z_dx2[1:-1, 1:-1] = (z[1:-1, 2:] + z[1:-1, :-2] - 2.0 * z[1:-1, 1:-1]) / (dx[1:-1] ** 2)
    d2z_dy2[1:-1, 1:-1] = (z[:-2, 1:-1] + z[2:, 1:-1] - 2.0 * z[1:-1, 1:-1]) / (dy ** 2)
    curvature = -(d2z_dx2 + d2z_dy2) * 100.0

    return {
        "elevation": dem,
        "slope_deg": slope_deg.astype(np.float32),
        "aspect_deg": aspect_deg.astype(np.float32),
        "aspect_northness": aspect_northness.astype(np.float32),
        "aspect_eastness": aspect_eastness.astype(np.float32),
        "curvature": curvature.astype(np.float32),
        "valid_mask": valid_mask,
    }


def aggregate_to_modeling_grid(
    features: Dict[str, np.ndarray],
    meghalaya_mask: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    factor: int = 3,
) -> Dict[str, np.ndarray]:
    """Aggregate native 30 m features to an intermediate 90 m (3-arc-second) landslide modeling grid."""
    H, W = features["elevation"].shape
    H_sub = H // factor
    W_sub = W // factor

    # Truncate arrays to multiple of factor
    z = features["elevation"][: H_sub * factor, : W_sub * factor]
    slope = features["slope_deg"][: H_sub * factor, : W_sub * factor]
    northness = features["aspect_northness"][: H_sub * factor, : W_sub * factor]
    eastness = features["aspect_eastness"][: H_sub * factor, : W_sub * factor]
    curv = features["curvature"][: H_sub * factor, : W_sub * factor]
    m_mask = meghalaya_mask[: H_sub * factor, : W_sub * factor]
    v_mask = features["valid_mask"][: H_sub * factor, : W_sub * factor]

    # Reshape for block reduction (H_sub, factor, W_sub, factor)
    def block_reduce_mean(arr: np.ndarray) -> np.ndarray:
        return arr.reshape(H_sub, factor, W_sub, factor).mean(axis=(1, 3)).astype(np.float32)

    def block_reduce_max(arr: np.ndarray) -> np.ndarray:
        return arr.reshape(H_sub, factor, W_sub, factor).max(axis=(1, 3)).astype(np.float32)

    def block_reduce_min(arr: np.ndarray) -> np.ndarray:
        return arr.reshape(H_sub, factor, W_sub, factor).min(axis=(1, 3)).astype(np.float32)

    def block_reduce_std(arr: np.ndarray) -> np.ndarray:
        return arr.reshape(H_sub, factor, W_sub, factor).std(axis=(1, 3)).astype(np.float32)

    z_mean = block_reduce_mean(z)
    z_min = block_reduce_min(z)
    z_max = block_reduce_max(z)
    relief = z_max - z_min

    slope_mean = block_reduce_mean(slope)
    slope_max = block_reduce_max(slope)
    slope_std = block_reduce_std(slope)

    northness_mean = block_reduce_mean(northness)
    eastness_mean = block_reduce_mean(eastness)
    curv_mean = block_reduce_mean(curv)

    mask_mean = block_reduce_mean(m_mask.astype(np.float32))
    state_mask_sub = (mask_mean >= 0.5).astype(np.uint8)
    valid_sub = (block_reduce_mean(v_mask.astype(np.float32)) >= 0.5).astype(np.uint8)

    lats_sub = lats[: H_sub * factor : factor]
    lons_sub = lons[: W_sub * factor : factor]

    return {
        "elevation_mean": z_mean,
        "elevation_min": z_min,
        "elevation_max": z_max,
        "relief": relief,
        "slope_mean": slope_mean,
        "slope_max": slope_max,
        "slope_std": slope_std,
        "aspect_northness": northness_mean,
        "aspect_eastness": eastness_mean,
        "curvature_mean": curv_mean,
        "meghalaya_mask": state_mask_sub,
        "valid_mask": valid_sub,
        "lats": lats_sub,
        "lons": lons_sub,
    }


def aggregate_to_weather_grid(
    features: Dict[str, np.ndarray],
    meghalaya_mask: np.ndarray,
    lats_native: np.ndarray,
    lons_native: np.ndarray,
    weather_sample_nc: Path,
) -> Dict[str, np.ndarray]:
    """Aggregate terrain morphometry onto the coarse 0.1-degree weather grid (25 x 40)."""
    with h5py.File(weather_sample_nc, "r") as hf:
        w_lats = hf["lat"][:]  # shape (25,) e.g. 24.5 to 26.9
        w_lons = hf["lon"][:]  # shape (40,) e.g. 89.5 to 93.4

    z_mean = np.zeros((len(w_lats), len(w_lons)), dtype=np.float32)
    z_max = np.zeros((len(w_lats), len(w_lons)), dtype=np.float32)
    relief = np.zeros((len(w_lats), len(w_lons)), dtype=np.float32)
    slope_mean = np.zeros((len(w_lats), len(w_lons)), dtype=np.float32)
    slope_max = np.zeros((len(w_lats), len(w_lons)), dtype=np.float32)
    slope_std = np.zeros((len(w_lats), len(w_lons)), dtype=np.float32)
    tri = np.zeros((len(w_lats), len(w_lons)), dtype=np.float32)
    coverage = np.zeros((len(w_lats), len(w_lons)), dtype=np.float32)

    half_cell = 0.05  # half of 0.1 degree

    dem = features["elevation"].astype(np.float32)
    slope = features["slope_deg"]

    for i, lat_c in enumerate(w_lats):
        lat_min = lat_c - half_cell
        lat_max = lat_c + half_cell
        lat_idx = np.where((lats_native >= lat_min) & (lats_native < lat_max))[0]
        if len(lat_idx) == 0:
            continue

        for j, lon_c in enumerate(w_lons):
            lon_min = lon_c - half_cell
            lon_max = lon_c + half_cell
            lon_idx = np.where((lons_native >= lon_min) & (lons_native < lon_max))[0]
            if len(lon_idx) == 0:
                continue

            sub_z = dem[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
            sub_slope = slope[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
            sub_mask = meghalaya_mask[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]

            if sub_z.size > 0:
                z_mean[i, j] = float(np.mean(sub_z))
                z_max[i, j] = float(np.max(sub_z))
                relief[i, j] = float(np.max(sub_z) - np.min(sub_z))
                slope_mean[i, j] = float(np.mean(sub_slope))
                slope_max[i, j] = float(np.max(sub_slope))
                slope_std[i, j] = float(np.std(sub_slope))
                # Topographic Ruggedness Index: root mean square of slope
                tri[i, j] = float(np.sqrt(np.mean(sub_slope**2)))
                coverage[i, j] = float(np.mean(sub_mask))

    return {
        "elevation_mean": z_mean,
        "elevation_max": z_max,
        "relief": relief,
        "slope_mean": slope_mean,
        "slope_max": slope_max,
        "slope_std": slope_std,
        "terrain_ruggedness_index": tri,
        "meghalaya_coverage": coverage,
        "lat": w_lats,
        "lon": w_lons,
    }


def compute_antecedent_rainfall(
    weather_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Compute 6h, 12h, 24h, 72h, and 7-day antecedent rainfall accumulation from IMERG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(weather_dir / "aligned_weather_mergir_202507*.nc4")))
    if not files:
        raise FileNotFoundError(f"No aligned weather files found in {weather_dir}")

    precip_list = []
    ref_times = []
    lats, lons = None, None

    for fpath in files:
        with h5py.File(fpath, "r") as hf:
            if lats is None:
                lats = hf["lat"][:]
                lons = hf["lon"][:]
            precip_list.append(hf["input_precip"][:])
            ref_times.extend([s.decode() if isinstance(s, bytes) else str(s) for s in hf["reference_time_iso"][:]])

    precip_all = np.concatenate(precip_list, axis=0)  # (336, 25, 40) in mm/hr
    # Accumulation per half-hour slot: Rate * 0.5 hr
    accum_slot = precip_all * 0.5  # mm

    T, H, W = accum_slot.shape

    # Durations in number of 30-min intervals
    intervals_map = {
        "accum_6h": 12,
        "accum_12h": 24,
        "accum_24h": 48,
        "accum_72h": 144,
        "accum_7d": 336,
    }

    accum_features: Dict[str, np.ndarray] = {}
    for name, n_slots in intervals_map.items():
        arr = np.full((T, H, W), np.nan, dtype=np.float32)
        for t in range(n_slots - 1, T):
            arr[t] = np.sum(accum_slot[t - n_slots + 1 : t + 1], axis=0)
        accum_features[name] = arr

    # Save to NetCDF
    out_file = output_dir / "antecedent_rainfall_july2025.nc4"
    with h5py.File(out_file, "w") as hf:
        hf.create_dataset("lat", data=lats)
        hf.create_dataset("lon", data=lons)
        hf.create_dataset("reference_time_iso", data=[s.encode("utf-8") for s in ref_times])
        for name, arr in accum_features.items():
            ds = hf.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
            ds.attrs["units"] = "mm"
            ds.attrs["description"] = f"Antecedent precipitation accumulation ({name})"

    # Summary statistics at peak time (July 7 end)
    summary = {
        "total_timesteps": T,
        "time_span": f"{ref_times[0]} to {ref_times[-1]}",
        "peak_accumulations_mm": {
            "max_6h": round(float(np.nanmax(accum_features["accum_6h"])), 1),
            "max_12h": round(float(np.nanmax(accum_features["accum_12h"])), 1),
            "max_24h": round(float(np.nanmax(accum_features["accum_24h"])), 1),
            "max_72h": round(float(np.nanmax(accum_features["accum_72h"])), 1),
            "max_7d": round(float(np.nanmax(accum_features["accum_7d"])), 1),
        },
        "output_file": str(out_file),
    }

    print(f"Antecedent rainfall computed: 7d peak={summary['peak_accumulations_mm']['max_7d']} mm")
    print(f"Saved to: {out_file}")
    return summary


def plot_terrain_and_events(
    grid_90m: Dict[str, np.ndarray],
    events_json_path: Path,
    output_png: Path,
) -> None:
    """Generate professional 6-panel terrain morphometry and landslide distribution figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=200)

    lats = grid_90m["lats"]
    lons = grid_90m["lons"]
    extent = [lons[0], lons[-1], lats[-1], lats[0]]  # [left, right, bottom, top]

    # Load historical landslide events
    events = []
    if events_json_path.exists():
        with open(events_json_path, "r", encoding="utf-8") as f:
            ev_data = json.load(f)
            events = ev_data.get("meghalaya_events_list", [])

    ev_lons = [float(e["lon"]) for e in events if e.get("lon")]
    ev_lats = [float(e["lat"]) for e in events if e.get("lat")]

    # 1. Elevation (SRTM 90m)
    ax = axes[0, 0]
    elev = grid_90m["elevation_mean"].copy()
    elev[grid_90m["meghalaya_mask"] == 0] = np.nan
    im = ax.imshow(elev, extent=extent, cmap="terrain", origin="upper", vmin=0, vmax=2000)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04, label="Elevation (m)")
    ax.contour(grid_90m["meghalaya_mask"], levels=[0.5], extent=extent, colors=["black"], linewidths=[1.2])
    ax.scatter(ev_lons, ev_lats, color="red", s=18, edgecolors="white", linewidths=0.6, label="NASA Landslide Events")
    ax.set_title("SRTM Elevation & Historical Landslides", fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    # 2. Slope (degrees)
    ax = axes[0, 1]
    slope = grid_90m["slope_mean"].copy()
    slope[grid_90m["meghalaya_mask"] == 0] = np.nan
    im = ax.imshow(slope, extent=extent, cmap="magma", origin="upper", vmin=0, vmax=45)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04, label="Mean Slope (deg)")
    ax.contour(grid_90m["meghalaya_mask"], levels=[0.5], extent=extent, colors=["black"], linewidths=[1.0])
    ax.set_title("Hillslope Gradient (Horn 1981)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")

    # 3. Maximum Sub-Pixel Slope (Initiation zones)
    ax = axes[0, 2]
    slope_max = grid_90m["slope_max"].copy()
    slope_max[grid_90m["meghalaya_mask"] == 0] = np.nan
    im = ax.imshow(slope_max, extent=extent, cmap="inferno", origin="upper", vmin=0, vmax=60)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04, label="Max Sub-Pixel Slope (deg)")
    ax.contour(grid_90m["meghalaya_mask"], levels=[0.5], extent=extent, colors=["black"], linewidths=[1.0])
    ax.set_title("Maximum Slope (Failure Initiation Escarpments)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")

    # 4. Aspect Northness
    ax = axes[1, 0]
    northness = grid_90m["aspect_northness"].copy()
    northness[grid_90m["meghalaya_mask"] == 0] = np.nan
    im = ax.imshow(northness, extent=extent, cmap="coolwarm", origin="upper", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04, label="Aspect Northness (cos theta)")
    ax.contour(grid_90m["meghalaya_mask"], levels=[0.5], extent=extent, colors=["black"], linewidths=[1.0])
    ax.set_title("Aspect Northness (Solar/Monsoon Exposure)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")

    # 5. Total Laplacian Curvature
    ax = axes[1, 1]
    curv = grid_90m["curvature_mean"].copy()
    curv[grid_90m["meghalaya_mask"] == 0] = np.nan
    # Clip extreme outliers for visualization
    curv_clipped = np.clip(curv, -15.0, 15.0)
    im = ax.imshow(curv_clipped, extent=extent, cmap="bwr", origin="upper", vmin=-15, vmax=15)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04, label="Curvature (1/m x 100)")
    ax.contour(grid_90m["meghalaya_mask"], levels=[0.5], extent=extent, colors=["black"], linewidths=[1.0])
    ax.set_title("Terrain Curvature (Blue=Valleys, Red=Ridges)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")

    # 6. Landslide Inventory Audit & Scientific Synthesis
    ax = axes[1, 2]
    ax.axis("off")
    diag_text = (
        "STEP 8: LANDSLIDE TERRAIN & INVENTORY AUDIT\n"
        "-----------------------------------------------------\n"
        "Study Area: Meghalaya, Northeast India\n"
        "DEM Source: NASA LP DAAC SRTM 30 m (SRTMGL1_NC.003)\n"
        "Tiles: 8 tiles (N25E089-N25E092, N26E089-N26E092)\n"
        "Boundary: geoBoundaries India ADM1 (12,517 vertices)\n"
        "Resolution: Native 30 m, Modeling 90 m, Coarse 0.1 deg\n\n"
        "TERRAIN MORPHOMETRY (MEGHALAYA AOI):\n"
        f"  - Elevation: {np.nanmin(elev):.0f} m to {np.nanmax(elev):.0f} m (Peak: Shillong Plateau)\n"
        f"  - Mean Slope: {np.nanmean(slope):.1f} deg (Max: {np.nanmax(slope_max):.1f} deg)\n"
        "  - Escarpment zones: South Garo/Khasi southern canyons\n\n"
        "LANDSLIDE INVENTORY AUDIT (NASA GLC / COOLR):\n"
        f"  - Total Historical Events in Meghalaya: {len(events)}\n"
        "  - Rainfall-Triggered Events: 55 (continuous rain/downpour)\n"
        "  - Positional Accuracy: 1 km (14), 5 km (25), 10-50 km (28)\n\n"
        "SCIENTIFIC FEASIBILITY ASSESSMENT:\n"
        "  * A daily 24-hr supervised prediction model across\n"
        "    Meghalaya is NOT scientifically feasible with 67\n"
        "    multi-year historical events (severe sparsity/leakage).\n"
        "  * Scientific Best Practice: Construct a Static Landslide\n"
        "    Susceptibility Model (LSM) coupled with empirical\n"
        "    antecedent rainfall trigger curves (NASA LHASA approach).\n"
        "  * Models must NOT claim next-day forecasting skill\n"
        "    without continuous event-monitoring networks."
    )
    ax.text(0.02, 0.98, diag_text, transform=ax.transAxes, fontsize=8.5,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.95))

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"Terrain & Landslide visualization saved to: {output_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Process SRTM DEM and Engineer Landslide Terrain Features")
    parser.add_argument("--dem-dir", type=str, default="data/raw/dem", help="Path to SRTM NetCDF tiles")
    parser.add_argument("--boundary", type=str, default="data/raw/boundary/meghalaya_boundary.geojson", help="Path to boundary GeoJSON")
    parser.add_argument("--weather-dir", type=str, default="data/processed/weather", help="Path to aligned weather NetCDF files")
    parser.add_argument("--output-dir", type=str, default="data/processed/terrain", help="Output directory for terrain rasters")
    parser.add_argument("--landslide-out", type=str, default="outputs/landslide", help="Output directory for reports and figures")
    args = parser.parse_args()

    out_terrain = Path(args.output_dir)
    out_landslide = Path(args.landslide_out)
    out_terrain.mkdir(parents=True, exist_ok=True)
    out_landslide.mkdir(parents=True, exist_ok=True)

    print("Step 1: Assembling 8-tile regional mosaic from NASA SRTM 30 m DEM...")
    mosaic, lats_full, lons_full = build_mosaic_from_tiles(Path(args.dem_dir))

    print(f"Step 2: Cropping mosaic to Meghalaya AOI ({AOI_BOUNDS['min_lat']}-{AOI_BOUNDS['max_lat']}N, {AOI_BOUNDS['min_lon']}-{AOI_BOUNDS['max_lon']}E)...")
    dem_aoi, lats_aoi, lons_aoi = crop_to_aoi(mosaic, lats_full, lons_full)
    print(f"  Cropped Native 30 m Grid Shape: {dem_aoi.shape}")

    print("Step 3: Rasterizing authoritative Meghalaya state boundary...")
    state_mask = rasterize_boundary_mask(Path(args.boundary), lats_aoi, lons_aoi)
    print(f"  Meghalaya State Pixels: {state_mask.sum()} / {state_mask.size} ({state_mask.mean()*100:.1f}%)")

    print("Step 4: Computing terrain morphometry (slope, aspect, curvature)...")
    morph_features = compute_terrain_morphometry(dem_aoi, lats_aoi, lons_aoi)

    print("Step 5: Aggregating to 90 m (3-arc-second) Regional Landslide Modeling Grid...")
    grid_90m = aggregate_to_modeling_grid(morph_features, state_mask, lats_aoi, lons_aoi, factor=3)

    # Save 90m Regional Modeling NetCDF
    nc_90m_path = out_terrain / "meghalaya_terrain_features_90m.nc4"
    print(f"  Saving 90 m terrain features to {nc_90m_path}...")
    with h5py.File(nc_90m_path, "w") as hf:
        hf.create_dataset("lat", data=grid_90m["lats"])
        hf.create_dataset("lon", data=grid_90m["lons"])
        for k in ["elevation_mean", "elevation_min", "elevation_max", "relief",
                  "slope_mean", "slope_max", "slope_std",
                  "aspect_northness", "aspect_eastness", "curvature_mean"]:
            hf.create_dataset(k, data=grid_90m[k], compression="gzip", compression_opts=4)
        hf.create_dataset("meghalaya_mask", data=grid_90m["meghalaya_mask"], compression="gzip")
        hf.create_dataset("valid_mask", data=grid_90m["valid_mask"], compression="gzip")

    print("Step 6: Aggregating terrain features onto aligned 0.1-degree weather grid...")
    sample_weather_nc = sorted(Path(args.weather_dir).glob("aligned_weather_mergir_202507*.nc4"))[0]
    weather_terrain = aggregate_to_weather_grid(morph_features, state_mask, lats_aoi, lons_aoi, sample_weather_nc)

    nc_weather_grid_path = out_terrain / "meghalaya_terrain_aligned_weather_grid.nc4"
    with h5py.File(nc_weather_grid_path, "w") as hf:
        hf.create_dataset("lat", data=weather_terrain["lat"])
        hf.create_dataset("lon", data=weather_terrain["lon"])
        for k in ["elevation_mean", "elevation_max", "relief", "slope_mean", "slope_max",
                  "slope_std", "terrain_ruggedness_index", "meghalaya_coverage"]:
            hf.create_dataset(k, data=weather_terrain[k], compression="gzip")

    print(f"  Saved coarse aligned terrain grid to: {nc_weather_grid_path}")

    print("Step 7: Computing antecedent rainfall accumulation features from IMERG...")
    antecedent_summary = compute_antecedent_rainfall(Path(args.weather_dir), Path("data/processed/landslide"))

    print("Step 8: Generating publication-grade terrain & landslide visualization...")
    plot_terrain_and_events(
        grid_90m=grid_90m,
        events_json_path=out_landslide / "landslide_inventory_audit.json",
        output_png=out_landslide / "meghalaya_terrain_and_events.png",
    )

    # Generate metadata inventory report
    meta_report = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline_step": "Step 8: Landslide Terrain & Antecedent Rainfall Feature Engineering",
            "study_area": "Meghalaya, India",
        },
        "datasets_generated": {
            "terrain_90m": str(nc_90m_path),
            "terrain_weather_grid": str(nc_weather_grid_path),
            "antecedent_rainfall": antecedent_summary["output_file"],
            "visualization": str(out_landslide / "meghalaya_terrain_and_events.png"),
            "landslide_events_csv": str(out_landslide / "meghalaya_landslide_events.csv"),
            "landslide_inventory_audit": str(out_landslide / "landslide_inventory_audit.json"),
        },
        "terrain_statistics": {
            "elevation_range_m": [float(np.min(grid_90m["elevation_min"])), float(np.max(grid_90m["elevation_max"]))],
            "slope_range_deg": [float(np.min(grid_90m["slope_mean"])), float(np.max(grid_90m["slope_max"]))],
            "meghalaya_coverage_pixels_90m": int(grid_90m["meghalaya_mask"].sum()),
        },
        "antecedent_rainfall_summary": antecedent_summary,
    }

    report_path = out_landslide / "terrain_and_landslide_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(meta_report, f, indent=2)

    print(f"\nAll Step 8 processing completed successfully. Summary saved to: {report_path}")


if __name__ == "__main__":
    main()
