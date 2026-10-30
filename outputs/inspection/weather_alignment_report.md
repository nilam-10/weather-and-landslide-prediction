# Weather Dataset Alignment & Verification Report

**Output NetCDF-4**: `aligned_weather_20250201.nc4` (58626 bytes)  
**Study Region**: Meghalaya, India (`(89.5, 24.5, 93.5, 27.0)`)  
**Common Grid**: 25 × 40 cells (0.1° resolution)  
**Rainfall Units**: `mm/hr` | **Temperature Units**: `K`  

---

## 1. Aligned Time Steps & Anti-Leakage Audit

| Slot | Reference Time (UTC) | Satellite Status | Input Rain Interval (UTC) | Margin (s) | Future Leakage |
|---|---|---|---|---|---|
| `00:15` | `00:42:30` | Available (Calibrated TIR1) | `00:00 – 00:30` | 750s | **PASSED** |
| `00:45` | `01:12:30` | Available (Calibrated TIR1) | `00:30 – 01:00` | 750s | **PASSED** |
| `01:15` | `01:42:29` | Available (Calibrated TIR1) | `01:00 – 01:30` | 749s | **PASSED** |
| `01:45` | `02:12:30` | Available (Calibrated TIR1) | `01:30 – 02:00` | 750s | **PASSED** |
| `02:15` | `02:42:30` | Available (Calibrated TIR1) | `02:00 – 02:30` | 750s | **PASSED** |
| `02:45` | `03:12:30` | **GAP (Masked NaN)** | `02:30 – 03:00` | 750s | **PASSED** |
| `03:15` | `03:42:30` | **GAP (Masked NaN)** | `03:00 – 03:30` | 750s | **PASSED** |
| `03:45` | `04:12:30` | **GAP (Masked NaN)** | `03:30 – 04:00` | 750s | **PASSED** |
| `04:15` | `04:42:30` | Available (Calibrated TIR1) | `04:00 – 04:30` | 750s | **PASSED** |

---

## 2. Integrity & Gap Audit Summary

- **Total Aligned Steps**: 9
- **Available Satellite Observations**: 6
- **Preserved Gap Intervals**: 3 (`02:45, 03:15, 03:45 UTC`)
- **Future Information Leakage**: **ZERO (VERIFIED)**
- **Synthetic Observations Fabricated**: **0 (None)**
