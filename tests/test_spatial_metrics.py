"""Automated Unit Tests for Fractions Skill Score (FSS) and Spatial Verification."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.spatial_metrics import SpatialVerificationEngine


def test_km_to_grid_radius_mapping():
    """Verify physical km to grid cell radius mapping."""
    eng = SpatialVerificationEngine
    assert eng.km_to_grid_radius(0.0) == 0
    assert eng.km_to_grid_radius(10.0, grid_km=10.55) == 1
    assert eng.km_to_grid_radius(20.0, grid_km=10.55) == 2
    assert eng.km_to_grid_radius(30.0, grid_km=10.55) == 3
    assert eng.km_to_grid_radius(50.0, grid_km=10.55) == 5


def test_fss_perfect_forecast():
    """Verify that identical non-empty fields yield FSS = 1.0."""
    obs = np.zeros((25, 40), dtype=np.float32)
    obs[10:15, 15:25] = 10.0  # Convective block
    fcst = obs.copy()

    res = SpatialVerificationEngine.compute_single_fss(obs, fcst, threshold=5.0, radius_cells=1)
    assert res["fss"] == 1.0
    assert res["fbs"] == 0.0
    assert not res["is_zero_event"]


def test_fss_zero_forecast_with_events():
    """Verify that a forecast with no rain when rain occurs yields FSS = 0.0."""
    obs = np.zeros((25, 40), dtype=np.float32)
    obs[10:15, 15:25] = 10.0
    fcst = np.zeros((25, 40), dtype=np.float32)

    res = SpatialVerificationEngine.compute_single_fss(obs, fcst, threshold=5.0, radius_cells=1)
    assert res["fss"] == 0.0
    assert not res["is_zero_event"]


def test_fss_pure_false_alarm():
    """Verify that predicting rain when none occurs yields FSS = 0.0."""
    obs = np.zeros((25, 40), dtype=np.float32)
    fcst = np.zeros((25, 40), dtype=np.float32)
    fcst[10:15, 15:25] = 10.0

    res = SpatialVerificationEngine.compute_single_fss(obs, fcst, threshold=5.0, radius_cells=1)
    assert res["fss"] == 0.0
    assert not res["is_zero_event"]


def test_fss_both_empty_zero_case():
    """Verify zero-event handling when neither forecast nor observation has rain."""
    obs = np.zeros((25, 40), dtype=np.float32)
    fcst = np.zeros((25, 40), dtype=np.float32)

    res = SpatialVerificationEngine.compute_single_fss(obs, fcst, threshold=5.0, radius_cells=1)
    assert res["fss"] == 1.0
    assert res["is_zero_event"]
    assert res["fbs"] == 0.0
    assert res["fbs_worst"] == 0.0


def test_fss_displacement_properties():
    """Verify that FSS responds logically to storm displacement."""
    eng = SpatialVerificationEngine

    # Ground truth: storm at (12, 18)
    obs = np.zeros((25, 40), dtype=np.float32)
    obs[11:14, 17:20] = 8.0  # 3x3 convective core

    # Shifted forecasts: 0, 1, 2, 4 cells shift in x direction
    fss_by_shift = []
    for shift in [0, 1, 2, 4]:
        fcst = np.zeros((25, 40), dtype=np.float32)
        fcst[11:14, (17 + shift) : (20 + shift)] = 8.0
        # Evaluate at radius = 2 cells (~20 km)
        res = eng.compute_single_fss(obs, fcst, threshold=5.0, radius_cells=2)
        fss_by_shift.append(res["fss"])

    # FSS must strictly decrease as spatial displacement increases
    assert fss_by_shift[0] == 1.0  # Perfect
    assert fss_by_shift[0] > fss_by_shift[1] > fss_by_shift[2] >= fss_by_shift[3]

    # For a fixed shift (2 cells), FSS must increase monotonically with neighborhood radius
    fcst_shift2 = np.zeros((25, 40), dtype=np.float32)
    fcst_shift2[11:14, 19:22] = 8.0

    fss_by_radius = []
    for r in [0, 1, 2, 3, 5]:
        res = eng.compute_single_fss(obs, fcst_shift2, threshold=5.0, radius_cells=r)
        fss_by_radius.append(res["fss"])

    # At r=0 (pointwise), overlapping cells = 1 col out of 3 -> low score
    # As radius expands, FSS must increase monotonically toward 1.0
    assert fss_by_radius[0] < fss_by_radius[1] <= fss_by_radius[2] <= fss_by_radius[3] <= fss_by_radius[4]


def test_fss_mask_handling():
    """Verify that invalid/missing pixels are masked and do not distort scores."""
    eng = SpatialVerificationEngine
    obs = np.zeros((25, 40), dtype=np.float32)
    fcst = np.zeros((25, 40), dtype=np.float32)
    mask = np.ones((25, 40), dtype=np.float32)

    obs[10:14, 15:19] = 10.0
    fcst[10:14, 15:19] = 10.0

    # Put heavy rain in a region where mask is 0
    obs[0:5, 0:5] = 20.0
    fcst[0:5, 0:5] = 0.0  # mismatch in invalid region!
    mask[0:5, 0:5] = 0.0  # marked invalid

    res = eng.compute_single_fss(obs, fcst, threshold=5.0, radius_cells=1, mask=mask)
    # The mismatch in the masked region must be ignored, yielding FSS = 1.0
    assert res["fss"] == 1.0


def test_neighborhood_contingency():
    """Verify Fuzzy Neighborhood Contingency (Ebert 2008)."""
    eng = SpatialVerificationEngine
    obs = np.zeros((25, 40), dtype=np.float32)
    fcst = np.zeros((25, 40), dtype=np.float32)

    # Storm shifted by 1 cell
    obs[10, 15] = 10.0
    fcst[10, 16] = 10.0

    # At r=0 (strict pointwise): TP=0, FP=1, FN=1 -> CSI = 0.0
    res_r0 = eng.compute_neighborhood_contingency(obs, fcst, threshold=5.0, radius_cells=0)
    assert res_r0["neighborhood_csi"] == 0.0
    assert res_r0["neighborhood_pod"] == 0.0
    assert res_r0["neighborhood_far"] == 1.0

    # At r=1 (neighborhood 3x3): Storm is within 1 cell -> matched!
    res_r1 = eng.compute_neighborhood_contingency(obs, fcst, threshold=5.0, radius_cells=1)
    assert res_r1["neighborhood_csi"] == 1.0
    assert res_r1["neighborhood_pod"] == 1.0
    assert res_r1["neighborhood_far"] == 0.0


def test_circular_kernel_geometry():
    """Verify circular Euclidean kernel shapes and active cell counts."""
    eng = SpatialVerificationEngine
    k1 = eng.create_spatial_kernel(1, kernel_type="circular")
    assert k1.shape == (3, 3)
    # Distance to center: (0,0)->0, cardinals->1, diagonals->sqrt(2)>1 => excluded
    assert int(np.sum(k1)) == 5

    k2 = eng.create_spatial_kernel(2, kernel_type="circular")
    assert k2.shape == (5, 5)
    # Points with x^2 + y^2 <= 4:
    # y=0: x in {-2,-1,0,1,2} (5)
    # y=+-1: x in {-1,0,1} (3*2=6)
    # y=+-2: x=0 (1*2=2) => total 13
    assert int(np.sum(k2)) == 13

    k3 = eng.create_spatial_kernel(3, kernel_type="circular")
    assert k3.shape == (7, 7)
    # Points with x^2 + y^2 <= 9: total 29
    assert int(np.sum(k3)) == 29

    k5 = eng.create_spatial_kernel(5, kernel_type="circular")
    assert k5.shape == (11, 11)
    # Points with x^2 + y^2 <= 25: total 81
    assert int(np.sum(k5)) == 81


def test_fss_denominator_and_hand_calculated_values():
    """Verify FSS and its denominator against an exact hand-calculated analytic field."""
    eng = SpatialVerificationEngine
    # Create 5x5 domain with all cells valid
    obs = np.zeros((5, 5), dtype=np.float32)
    fcst = np.zeros((5, 5), dtype=np.float32)

    # Event threshold = 5.0
    # Center cell is an event in obs
    obs[2, 2] = 10.0
    # Adjacent cell (2, 3) is an event in fcst (1-cell shift)
    fcst[2, 3] = 10.0

    # With k=0 (pointwise):
    # obs fraction field: 1 at (2,2), 0 elsewhere
    # fcst fraction field: 1 at (2,3), 0 elsewhere
    # diff_sq has two 1s: sum(diff_sq) = 2.0, FBS = 2/25
    # sum_sq has two 1s: sum(sum_sq) = 2.0, FBS_worst = 2/25
    # FSS = 1 - 2/2 = 0.0
    res_k0 = eng.compute_single_fss(obs, fcst, threshold=5.0, radius_cells=0)
    assert res_k0["fss"] == 0.0
    assert abs(res_k0["fbs"] - (2.0 / 25.0)) < 1e-6
    assert abs(res_k0["fbs_worst"] - (2.0 / 25.0)) < 1e-6

    # With k=1 (square 3x3 window):
    # Center cell (2,2) has 9 neighbors (all valid). Both (2,2) and (2,3) fall in the 3x3 window of cell (2,2)
    # Both fields have 1 event in the domain.
    # We verify that FBS <= FBS_worst and FSS > 0.0
    res_k1 = eng.compute_single_fss(obs, fcst, threshold=5.0, radius_cells=1, kernel_type="square")
    assert res_k1["fss"] > 0.0
    assert res_k1["fbs_worst"] > res_k1["fbs"]
    # Mathematical identity: FSS = 1 - FBS / FBS_worst
    expected_fss = 1.0 - (res_k1["fbs"] / res_k1["fbs_worst"])
    assert abs(res_k1["fss"] - expected_fss) < 1e-6


def test_boundary_normalization_correctness():
    """Verify that boundary cells are divided by the count of valid cells inside domain."""
    eng = SpatialVerificationEngine
    H, W = 10, 10
    binary_field = np.zeros((H, W), dtype=np.float32)
    # Place event at top-left corner (0, 0)
    binary_field[0, 0] = 1.0

    fractions, valid = eng.compute_fraction_field(binary_field, radius_cells=1, kernel_type="square")

    # Corner (0, 0) has only 4 neighbors in a 10x10 domain: (0,0), (0,1), (1,0), (1,1)
    # Event count is 1. Therefore fraction at (0, 0) MUST be exactly 1 / 4 = 0.25!
    # (If unnormalized by boundary, it would be 1 / 9 = 0.111, which would be a bug)
    assert abs(fractions[0, 0] - 0.25) < 1e-6

    # Edge cell (0, 5) has 6 neighbors: row 0 cols 4,5,6; row 1 cols 4,5,6
    # Interior cell (5, 5) has 9 neighbors: 3x3
    binary_center = np.zeros((H, W), dtype=np.float32)
    binary_center[5, 5] = 1.0
    fractions_c, _ = eng.compute_fraction_field(binary_center, radius_cells=1, kernel_type="square")
    assert abs(fractions_c[5, 5] - (1.0 / 9.0)) < 1e-6

