# External APIs, Data Providers & Environment Requirements Audit

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Component:** External Integrations & Infrastructure Configuration Audit  
**Location:** `outputs/experiments/landslide_model_comparison/EXTERNAL_API_AND_ENVIRONMENT_REQUIREMENTS.md`  
**Date:** October 30, 2026  

---

## 1. External Service & API Audit

This audit systematically checks all external APIs, data feeds, tile services, and credentials referenced in the codebase:

| External Service / Provider | Functionality Supplied | Code Location & Caller | Needed for Local Operation? | API Key / Token Required? | Env Variable Name (if applicable) | Licensing & Cost | Offline Behavior / Fallback |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **CARTO Dark Matter Map Tiles** | Base map visual styling & geographic basemap tiles | `frontend/index.html` (`L.tileLayer`) | **No** (Map loads blank dark background if offline) | **None** | None | Free / OpenStreetMap Attribution | If internet is disconnected, the Leaflet map displays a dark grey background while all weather and landslide polygons still render correctly. |
| **Leaflet CDN (unpkg.com)** | Client-side Leaflet 1.9.4 CSS and JS libraries | `frontend/index.html` (lines 8, 814) | **Only on first page load** (cached by browser) | **None** | None | Open Source (BSD-2-Clause) | If completely offline without browser cache, Leaflet CSS/JS can be bundled locally into `frontend/vendor/`. |
| **Google Fonts CDN** | Typography (Inter, JetBrains Mono) | `frontend/index.html` (line 10) | **No** | **None** | None | Free / SIL Open Font License | Browser cleanly falls back to system monospace and sans-serif fonts. |
| **NASA Earthdata (GES DISC)** | Historical GPM IMERG and MERGIR satellite granules | `scripts/download_imerg.py`, `scripts/download_mergir.py` | **NO for live application** (Data is already processed locally) | Yes (Earthdata Login for raw downloads only) | `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD` | Free for scientific research | **Not needed for runtime.** The 7-day monsoon archive is fully preprocessed into local `.nc4` files. |
| **ISRO MOSDAC (Meteorological Archive)** | INSAT-3D/3DR Satellite TIR and Sounder granules | `scripts/download_mosdac.py`, `scripts/inspect_insat.py` | **NO for live application** | Yes (MOSDAC Portal Login for raw downloads only) | `.mosdac_credentials.json` | Free for registered academic research | **Not needed for runtime.** Used solely in Phase 1 data exploration. |
| **USGS / NASA EarthExplorer** | SRTM 1 Arc-Second (30m/90m) Digital Elevation Model | `scripts/download_dem.py`, `scripts/process_terrain.py` | **NO for live application** | Yes (USGS EROS account for raw downloads only) | `EARTHDATA_LOGIN` | Public Domain / CC0 | **Not needed for runtime.** Terrain morphometry rasters are already processed locally in `meghalaya_terrain_features_90m.nc4`. |

---

## 2. Environment Variables & Security Hygiene

### Codebase Audit for Secrets & Keys:
1. **Zero Secret Hardcoding:** An exhaustive code scan confirmed that **no private API keys, Earthdata passwords, or personal access tokens are hardcoded** in `src/api/main.py`, `frontend/index.html`, or any model checkpoint loader.
2. **Local Operational Autonomy:** The application runs **100% self-contained locally**. To run the FastAPI server and the Leaflet dashboard, **zero environment variables and zero external API keys are required**.
3. **CORS Configuration:** In `src/api/main.py` (lines 317–323), CORS is configured with `CORSMiddleware(allow_origins=["*"])`, allowing seamless local communication between the frontend and backend on `http://127.0.0.1:8000`.
4. **Offline Capability:** If the laptop is disconnected from Wi-Fi, the backend runs at full speed on `http://127.0.0.1:8000/`. The only external assets are the CartoDB basemap tiles (which will simply appear as a solid dark background without interrupting model inference).

---

## 3. Mandatory Local Data Files Checklist

The live platform requires the following preprocessed local data files in `D:\Downloads\honours`:

| File Path | Description | File Size | Readiness Status |
| :--- | :--- | :---: | :---: |
| `data/processed/weather/aligned_weather_mergir_20250701.nc4` to `...20250707.nc4` | 7-day half-hourly weather cubes (336 steps) containing IMERG rain, MERGIR IR Tb, and validity mask. | ~18 MB each | **VERIFIED PRESENT** |
| `data/processed/terrain/meghalaya_terrain_features_90m.nc4` | High-resolution 90m SRTM DEM with 8 morphometric derivatives. | ~158 MB | **VERIFIED PRESENT** |
| `data/processed/landslide/meghalaya_dynamic_hazard_july2025.nc4` | Rolling 24h and 72h antecedent rainfall grids with LHASA 2.0 hazard tiers. | ~22.7 MB | **VERIFIED PRESENT** |
| `data/raw/boundary/meghalaya_boundary.geojson` | Official state boundary polygon for Meghalaya. | ~184 KB | **VERIFIED PRESENT** |
| `outputs/landslide/meghalaya_landslide_events.csv` | Catalog of verified rainfall-triggered landslide events ($N=38$). | ~44 KB | **VERIFIED PRESENT** |
| `outputs/models/phase3/convlstm_seed42.pt` | MultiTask ConvLSTM neural network checkpoint (161K params). | ~650 KB | **VERIFIED PRESENT** |
| `outputs/models/landslide_susceptibility/xgboost_spatial_cv.joblib` | Production XGBoost landslide susceptibility model. | ~109 KB | **VERIFIED PRESENT** |
| `outputs/models/landslide_susceptibility/random_forest_spatial_cv.joblib` | Production Random Forest landslide susceptibility model. | ~211 KB | **VERIFIED PRESENT** |
| `outputs/models/landslide_susceptibility/logistic_regression_spatial_cv.joblib` | Production Logistic Regression landslide baseline. | ~1.7 KB | **VERIFIED PRESENT** |
