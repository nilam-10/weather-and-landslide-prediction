# DATA EXPANSION PLAN: MULTI-YEAR SATELLITE MONSOON ARCHIVE FOR MEGHALAYA

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Component:** Remote Sensing Data Ingestion Pipeline  
**Target Domain:** Meghalaya Plateau ($24.5^\circ\text{N}-27.0^\circ\text{N}, 89.5^\circ\text{E}-93.5^\circ\text{E}$, $0.1^\circ$ Grid)  
**Author:** Senior Research Engineer in Remote Sensing & Spatiotemporal Deep Learning  
**Date:** October 29, 2026  
**Status:** Feasibility Verified; Architecture & Ingestion Protocols Specified  

---

## 1. DATA EXPANSION FEASIBILITY AUDIT

### 1.1 Local Archive Status
The current repository contains **49 active monsoon days** (2,399 raw IMERG granules, 2,345 raw MERGIR granules, totaling 2,352 half-hourly time slots):
* 2023: July 1 to July 14 (14 days, 672 half-hour slots)
* 2024: July 1 to July 14 (14 days, 672 half-hour slots)
* 2024: August 1 to August 14 (14 days, 672 half-hour slots)
* 2025: July 1 to July 7 (7 days, 336 half-hour slots)

All 49 local monsoon days have been preprocessed and aligned into standard NetCDF4 files in `data/processed/weather/aligned_weather_mergir_YYYYMMDD.nc4`.

### 1.2 The Expansion Target: Complete 4-Season Monsoon Archive
To expand the dataset for industrial-scale foundation nowcasting without modifying the spatial grid or physical units, the recommended target is to acquire **four complete southwest monsoon seasons (June 1 to September 30, 2021–2024)**.

| Parameter | Current Local Dataset | Proposed Expanded Archive |
| :--- | :--- | :--- |
| **Monsoon Seasons** | 4 partial episodes (July 2023, July/Aug 2024, July 2025) | 4 full seasons (June–Sept 2021, 2022, 2023, 2024) |
| **Total Calendar Days** | 49 active monsoon days | **488 monsoon days** ($4 \times 122\text{ days}$) |
| **Observation Slots** | 2,352 half-hourly slots | **23,424 half-hourly slots** |
| **Sliding Sequences ($T_{in}=4$)** | 1,764 sequences | **$\approx 21,500$ sequences** |
| **Storage Footprint (Aligned)** | $\approx 15\text{ MB}$ NetCDF4 | **$\approx 145\text{ MB}$ NetCDF4** |

---

## 2. EXACT PRODUCTS, ACCESS PROTOCOLS & DATA COMPATIBILITY

### 2.1 Satellite Precipitation: NASA GPM IMERG Final Run V07B
* **Product Identifier:** `GPM_3IMERGHH_07` (GPM IMERG Half-Hourly $0.1^\circ \times 0.1^\circ$ V07B)
* **DOI:** [10.5067/GPM/IMERG/3B-HH/07](https://doi.org/10.5067/GPM/IMERG/3B-HH/07)
* **Archive Host:** NASA Goddard Earth Sciences Data and Information Services Center (GES DISC)
* **Server Endpoint:** `https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/`
* **Access Protocol:** NASA Earthdata Login (URS) OAuth2 Bearer Tokens or `.netrc` authentication via CMR Granule Search API (`https://cmr.earthdata.nasa.gov/search/granules.json`).
* **Variable Extracted:** `/Grid/precipitation` (floating point rain rate in mm/hr).
* **Domain Extraction:** Spatial slicing along latitudes $[24.5^\circ\text{N}, 27.0^\circ\text{N}]$ (indices 1145 to 1170) and longitudes $[89.5^\circ\text{E}, 93.5^\circ\text{E}]$ (indices 2695 to 2735).

### 2.2 Satellite Infrared: NOAA/NASA NCEP Globally Merged IR (MERGIR)
* **Product Identifier:** `NCEP-GPM-MERGED-IR` (Globally Merged 4 km Half-Hourly 10.7 µm IR $T_b$)
* **DOI:** [10.5067/P4HZ5J0144ST](https://doi.org/10.5067/P4HZ5J0144ST)
* **Archive Host:** NASA GES DISC
* **Server Endpoint:** `https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/NCEP_GPM_MERGIR.1/`
* **Variable Extracted:** Binary 8-bit / 16-bit integer brightness temperature converted to physical Kelvin:
  $$T_b(\text{K}) = \text{raw\_value} + 75.0\text{ K}$$
* **Spatial Alignment:** Resampled from 4 km native projection to the IMERG $0.1^\circ$ grid using area-weighted bilinear interpolation.

---

## 3. STREAMING INGESTION & SUBSETTING RUNBOOK

Because downloading global IMERG and MERGIR grids would require over **2.4 Terabytes** of raw downloads, the ingestion pipeline utilizes **on-the-fly spatial subsetting** directly through NASA OPeNDAP or stream-range HTTP requests:

```
+-----------------------------------------------------------------------------+
|                     NASA GES DISC STREAMING INGESTION                       |
+-----------------------------------------------------------------------------+
| Global Granule (~15 MB) ---> [ HTTP Range Request / OPeNDAP Slicing ]       |
|                                                     |                       |
|                                                     v                       |
|                             Meghalaya Crop Only (25 x 40 x 3, ~3 KB/slot)   |
|                                                     |                       |
|                                                     v                       |
|                             Daily Consolidated NetCDF4 (~300 KB/day)        |
+-----------------------------------------------------------------------------+
```

### Reproducible Execution Script:
```powershell
# Ingest single or multi-year episodes once NASA credentials are authenticated:
python scripts/download_imerg.py --start-date 2022-06-01 --end-date 2022-09-30 --bbox 24.5 89.5 27.0 93.5
python scripts/download_mergir.py --start-date 2022-06-01 --end-date 2022-09-30 --bbox 24.5 89.5 27.0 93.5
python scripts/align_weather.py --start-date 2022-06-01 --end-date 2022-09-30
```

---

## 4. SCIENTIFIC BENEFIT FOR LONGER HORIZON NOWCASTING

1. **Severe Convective Event Sampling:** The current 38 training days contain approximately 56,000 heavy rainfall pixel instances ($P \ge 5.0\text{ mm/hr}$, $3.2\%$ prevalence). A 488-day archive increases positive instances to $> 700,000$, providing sufficient extreme-rainfall samples to train deep spatiotemporal models without parameter starvation.
2. **Seasonal Generalization:** Captures early-monsoon onset (June), peak active-break synoptic phases (July–August), and retreating monsoon cloud patterns (September).
3. **Multi-Horizon Predictability:** While single convective cell lifecycles remain 30–90 minutes, a 488-day archive enables the network to learn multi-day synoptic monsoon depressions propagating from the Bay of Bengal into the Meghalaya hills, supporting genuine predictability out to +360 minutes.
