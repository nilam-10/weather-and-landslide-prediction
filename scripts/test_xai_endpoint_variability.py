"""Verification script to test real computed variability and error handling on /explain/weather.
"""
import sys
from pathlib import Path
if "." not in sys.path:
    sys.path.insert(0, ".")

import pytest
from src.api.main import initialize_cache, cache, explain_weather_prediction
from fastapi import HTTPException

def run_tests():
    initialize_cache()
    print("=== TEST 1: Cell Location Variability ===")
    res_core = explain_weather_prediction(step=316, lead="30m", target_type="probability", method="integrated_gradients", row=5, col=23)
    res_east = explain_weather_prediction(step=316, lead="30m", target_type="probability", method="integrated_gradients", row=5, col=34)
    res_plat = explain_weather_prediction(step=316, lead="30m", target_type="probability", method="integrated_gradients", row=12, col=20)
    
    fx_core = res_core["model_output"]["fx"]
    fx_east = res_east["model_output"]["fx"]
    fx_plat = res_plat["model_output"]["fx"]
    
    print(f"Fx Core: {fx_core}, Fx East: {fx_east}, Fx Plateau: {fx_plat}")
    assert fx_core != fx_east != fx_plat, "Predictions must vary across cells!"
    assert res_core["grids"]["spatial_attribution_raw"] != res_plat["grids"]["spatial_attribution_raw"], "Attribution maps must vary!"
    print("-> PASS: Spatial variability confirmed.")

    print("\n=== TEST 2: Forecast Horizon Variability ===")
    res_30m = explain_weather_prediction(step=316, lead="30m", target_type="probability", method="integrated_gradients", row=5, col=23)
    res_180m = explain_weather_prediction(step=316, lead="180m", target_type="probability", method="integrated_gradients", row=5, col=23)
    print(f"Fx 30m: {res_30m['model_output']['fx']}, Fx 180m: {res_180m['model_output']['fx']}")
    assert res_30m["model_output"]["fx"] != res_180m["model_output"]["fx"], "Predictions must vary across leads!"
    assert res_30m["lead_time"] == "30m" and res_180m["lead_time"] == "180m"
    print("-> PASS: Horizon variability confirmed.")

    print("\n=== TEST 3: Target Output Head Variability (Probability vs QPF) ===")
    res_prob = explain_weather_prediction(step=316, lead="30m", target_type="probability", method="integrated_gradients", row=5, col=23)
    res_qpf = explain_weather_prediction(step=316, lead="30m", target_type="qpf", method="integrated_gradients", row=5, col=23)
    print(f"Prob target unit: {res_prob['target_unit']}, fx: {res_prob['model_output']['fx']}")
    print(f"QPF target unit: {res_qpf['target_unit']}, fx: {res_qpf['model_output']['fx']}")
    assert res_prob["target_unit"] != res_qpf["target_unit"]
    assert "mm/hr" in res_qpf["target_unit"]
    assert res_qpf["model_output"]["fx"] > 0
    print("-> PASS: Dual-head variability confirmed.")

    print("\n=== TEST 4: Method Variability (Integrated Gradients vs Occlusion) ===")
    res_ig = explain_weather_prediction(step=316, lead="30m", target_type="probability", method="integrated_gradients", row=5, col=23, m_steps=20)
    res_occ = explain_weather_prediction(step=316, lead="30m", target_type="probability", method="occlusion", row=5, col=23, patch_size=3)
    assert res_ig["method"] == "integrated_gradients"
    assert res_occ["method"] == "occlusion"
    assert "completeness_verified" in res_ig["completeness_proof"]
    assert res_ig["grids"]["spatial_attribution_raw"] != res_occ["grids"]["spatial_attribution_raw"]
    print("-> PASS: Method variability confirmed.")

    print("\n=== TEST 5: Geographic Coordinate Indexing ===")
    # Cell (lat=25.05, lon=91.85) should map to row 5, col 23
    res_geo = explain_weather_prediction(step=316, lead="30m", target_type="probability", method="integrated_gradients", lat=25.05, lon=91.85)
    assert res_geo["target_cell"]["row"] == 5 and res_geo["target_cell"]["col"] == 23
    print(f"Mapped lat=25.05, lon=91.85 -> row={res_geo['target_cell']['row']}, col={res_geo['target_cell']['col']}")
    print("-> PASS: Geographic coordinate indexing confirmed.")

    print("\n=== TEST 6: Error Handling ===")
    try:
        explain_weather_prediction(step=316, lead="30m", method="unsupported_method", row=5, col=23)
        assert False, "Should raise HTTPException for unknown method"
    except HTTPException as e:
        print(f"Caught expected error for unknown method: {e.detail}")
        assert e.status_code == 400

    try:
        explain_weather_prediction(step=9999, lead="30m", method="integrated_gradients", row=5, col=23)
        assert False, "Should raise HTTPException for out of bounds step"
    except HTTPException as e:
        print(f"Caught expected error for out of bounds step: {e.detail}")
        assert e.status_code == 400

    print("-> PASS: Error handling verified.")
    print("\n=== ALL VARIABILITY & API TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()
