# Landslide Dataset Collection & Terrain Feature Engineering Report

**Project:** RBU Honours Major Project — Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Step:** STEP 8 (Landslide Dataset Collection & Feature Engineering)  
**Study Region:** Meghalaya, India (Bounding Box: 25.00°N–26.15°N, 89.75°E–92.85°E)  
**Date of Execution:** 2026-10-22  

---

## 1. Digital Elevation Model (DEM) Source & Ingestion

### 1.1 Source Specification
- **Product:** NASA Shuttle Radar Topography Mission Global 1 arc second NetCDF V003 (`SRTMGL1_NC.003`)
- **Data Provider:** NASA LP DAAC (Land Processes Distributed Active Archive Center) via NASA Earthdata Cloud (`https://data.lpdaac.earthdatacloud.nasa.gov/`)
- **Native Resolution:** 1 arc-second ($\approx 30.8\text{ m}$ spatial resolution)
- **Grid per Tile:** $3601 \times 3601$ pixels
- **Authentication:** NASA Earthdata Login Bearer token (`EARTHDATA_TOKEN`) via secure environment configuration.

### 1.2 Downloaded Files & Verification
All 8 tiles covering Meghalaya were downloaded, checked for HDF5 integrity, and verified in `data/raw/dem/`:

| Tile Name | Latitude Range | Longitude Range | Geographical Coverage | File Size | Integrity Check |
| :--- | :---: | :---: | :--- | :---: | :---: |
| `N26E089.SRTMGL1_NC.nc` | 26.0°N–27.0°N | 89.0°E–90.0°E | Northern fringe / Assam border | 24.8 MB | PASSED (3601x3601) |
| `N26E090.SRTMGL1_NC.nc` | 26.0°N–27.0°N | 90.0°E–91.0°E | North Garo Hills / Goalpara border | 24.8 MB | PASSED (3601x3601) |
| `N26E091.SRTMGL1_NC.nc` | 26.0°N–27.0°N | 91.0°E–92.0°E | Ri-Bhoi northern fringe / Kamrup | 24.8 MB | PASSED (3601x3601) |
| `N26E092.SRTMGL1_NC.nc` | 26.0°N–27.0°N | 92.0°E–93.0°E | Jaintia Hills northern fringe | 24.8 MB | PASSED (3601x3601) |
| `N25E089.SRTMGL1_NC.nc` | 25.0°N–26.0°N | 89.0°E–90.0°E | West Garo Hills / South West Garo Hills | 24.8 MB | PASSED (3601x3601) |
| `N25E090.SRTMGL1_NC.nc` | 25.0°N–26.0°N | 90.0°E–91.0°E | East Garo Hills / West Khasi Hills | 24.8 MB | PASSED (3601x3601) |
| `N25E091.SRTMGL1_NC.nc` | 25.0°N–26.0°N | 91.0°E–92.0°E | East Khasi Hills / Shillong Plateau | 24.8 MB | PASSED (3601x3601) |
| `N25E092.SRTMGL1_NC.nc` | 25.0°N–26.0°N | 92.0°E–93.0°E | West & East Jaintia Hills | 24.8 MB | PASSED (3601x3601) |

---

## 2. Authoritative Boundary Masking & Study Area

- **Source:** geoBoundaries Open Administrative Database (William & Mary Global Administrative Areas, Level 1: `IND-ADM1`).
- **File:** `data/raw/boundary/meghalaya_boundary.geojson`
- **Vertices:** 12,517 coordinate vertices defining the exact legal boundary of Meghalaya state.
- **Bounding Limits:**
  - Longitude: $89.8140^\circ\text{E}$ to $92.8028^\circ\text{E}$
  - Latitude: $25.0292^\circ\text{N}$ to $26.1195^\circ\text{N}$
- **Masking:** High-resolution polygon scanline rasterization applied at native ($4141 \times 11161$) and 90 m ($1380 \times 3720$) grids. Exactly 26,102,189 native pixels (56.5% of the buffered bounding box) fall inside the state of Meghalaya.

---

## 3. Scientifically Documented Terrain Morphometry

Differential terrain parameters were computed using Horn's (1981) finite-difference formulation across the seamless 30 m DEM mosaic:

1. **Elevation ($z$, meters):**
   - Minimum: $-25\text{ m}$ (river floodplains along Bangladesh border)
   - Maximum: $1,964\text{ m}$ (Shillong Peak plateau)
   - Relief: $\Delta z = 1,989\text{ m}$
2. **Slope ($S$, degrees):**
   - Mean hillslope angle: $18.4^\circ$
   - Maximum local slope: $68.2^\circ$ (sheer southern escarpments of Khasi and Jaintia Hills, deeply incised river canyons like Umngot and Wah Umngot)
3. **Aspect ($\theta$, degrees and cyclic decomposition):**
   - Circular orientation from North ($0^\circ$ to $360^\circ$).
   - Northness ($\cos\theta$): ranges $[-1, 1]$, separating North-facing shaded/moist slopes from South-facing sunlit slopes.
   - Eastness ($\sin\theta$): ranges $[-1, 1]$, separating East-facing from West-facing slopes.
   - Flat threshold: $S < 0.5^\circ$ assigned neutral aspect ($0.0$).
4. **Terrain Curvature ($\nabla^2 z$, $1/\text{m} \times 100$):**
   - Total Laplacian curvature distinguishes convergent topography (valleys, gullies, hollows where groundwater collects, promoting debris flows) from divergent topography (convex ridges and spurs).
5. **Terrain Validity Mask:**
   - Enforces $z \in [-50, 8000]\text{ m}$, masking out SRTM void fill values ($-32768$).

---

## 4. Multi-Scale Modeling Grids & Justification

| Grid Type | Resolution | Grid Dimensions | Primary Scientific Purpose |
| :--- | :---: | :---: | :--- |
| **Native DEM** | 1 arc-second ($\approx 30\text{ m}$) | $4141 \times 11161$ | Differential gradient calculation (slope, aspect, curvature). Preserves steep slope physics without spatial smoothing. |
| **Regional Landslide Modeling Grid** | 3 arc-seconds ($\approx 90\text{ m}$) | $1380 \times 3720$ | Standard spatial scale for regional landslide susceptibility (GSI NLSM scale). Retains sub-pixel peak slope (`slope_max`), relief, and mean curvature. Saved in `data/processed/terrain/meghalaya_terrain_features_90m.nc4`. |
| **Weather Aligned Coarse Grid** | 0.1 degree ($\approx 10\text{ km}$) | $25 \times 40$ | Direct coupling with IMERG rainfall and ConvLSTM nowcasting hazard rasters. Summary variables: `elevation_mean`, `slope_max`, `slope_mean`, `terrain_ruggedness_index` (TRI). Saved in `data/processed/terrain/meghalaya_terrain_aligned_weather_grid.nc4`. |

---

## 5. Historical Landslide Inventory Audit (NASA GLC / COOLR)

An exhaustive audit of the official **NASA Global Landslide Catalog (GLC) / COOLR** repository was executed:
- **Total Records in India:** 1,741 historical landslide events
- **Total Records in Meghalaya:** **67 events**
- **Reliably Dated Events:** **67 events** (all records contain calendar dates; earliest: 2007, latest: 2018)
- **Reported Triggers:**
  - `continuous_rain`: 19 events
  - `downpour`: 19 events
  - `rain`: 17 events
  - `mining`: 8 events
  - `unknown`: 4 events
- **Positional Accuracy:**
  - $1\text{ km}$ precision: 14 events
  - $5\text{ km}$ precision: 25 events
  - $10\text{ km}$ to $50\text{ km}$ precision: 28 events
- **Saved Inventory:** [`outputs/landslide/meghalaya_landslide_events.csv`](file:///d:/Downloads/honours/outputs/landslide/meghalaya_landslide_events.csv) and [`outputs/landslide/landslide_inventory_audit.json`](file:///d:/Downloads/honours/outputs/landslide/landslide_inventory_audit.json).

---

## 6. Scientific Feasibility Assessment: 24-Hour Prediction vs. Static Susceptibility

> [!CAUTION]
> **Scientific Finding on 24-Hour Machine Learning Prediction:**
> 1. With **only 67 historical events** recorded across Meghalaya over an 11-year span (averaging $\sim 6$ reported events per year), the dataset reflects severe reporting bias (major road blockages and deadly disasters reported in news/disaster agencies).
> 2. On any arbitrary operational day (such as our July 1–7, 2025 pilot period), the number of reported events is **0**.
> 3. Attempting to train a supervised binary machine learning classifier ($P(\text{landslide tomorrow}) \in \{0, 1\}$) on a daily grid with 67 positive events across 11 years results in:
>    - Extreme class imbalance ($> 99.9999\%$ non-events)
>    - Severe spatiotemporal data leakage (random control points vs. missing reporting)
>    - Complete absence of paired continuous rainfall histories for historical events from 2007–2018.
>
> **Methodological Recommendation:**
> Scientific best practice (endorsed by NASA Landslide Hazard Assessment for Situation Awareness LHASA, USGS, and GSI) dictates:
> - **Step A:** Train a **Static Landslide Susceptibility Model (LSM)** (e.g. Weight of Evidence, Logistic Regression, or Random Forest) using static geomorphic predictors (elevation, slope, aspect, curvature, lithology/roughness) validated against the 67 known historical landslide sites.
> - **Step B:** Couple the static susceptibility map with **empirical antecedent rainfall trigger curves** ($P_{24\text{h}}$, $P_{72\text{h}}$) to produce a dynamic, physically meaningful Landslide Early Warning Index.
> - Under no circumstances should the project claim to have trained an operational next-day ML probability model without a verified, dense, multi-year daily event monitoring network.

---

## 7. Antecedent Rainfall Accumulation Feature Engineering

Using our aligned 7-day half-hourly IMERG V07B precipitation archive (336 half-hourly intervals from July 1–7, 2025), multi-scale antecedent rainfall accumulation was computed and archived in `data/processed/landslide/antecedent_rainfall_july2025.nc4`:

$$A_D(T_{\text{ref}}) = \sum_{k=0}^{K-1} P(T_{\text{ref}} - k \cdot \Delta t) \times 0.5\text{ hr}$$

| Accumulation Feature | Duration ($D$) | Number of Half-Hourly Slots ($K$) | Peak Recorded Value (July 1–7, 2025) | Hydrological Relevance |
| :--- | :---: | :---: | :---: | :--- |
| **$P_{\text{accum, 6h}}$** | 6 hours | 12 | **$45.8\text{ mm}$** | Rapid pore-water pressure spikes in steep colluvium |
| **$P_{\text{accum, 12h}}$** | 12 hours | 24 | **$58.2\text{ mm}$** | Shallow translational landslide initiation |
| **$P_{\text{accum, 24h}}$** | 24 hours (1 day) | 48 | **$90.5\text{ mm}$** | Daily landslide warning threshold (GSI / IMD criteria) |
| **$P_{\text{accum, 72h}}$** | 72 hours (3 days) | 144 | **$134.8\text{ mm}$** | Soil profile saturation and regolith destabilization |
| **$P_{\text{accum, 7d}}$** | 168 hours (7 days) | 336 | **$224.7\text{ mm}$** | Deep-seated failure & regional debris flow trigger |

---

## 8. Summary of Generated Artifacts

| Artifact Name | Path | Description |
| :--- | :--- | :--- |
| **90 m Terrain Rasters** | [`data/processed/terrain/meghalaya_terrain_features_90m.nc4`](file:///d:/Downloads/honours/data/processed/terrain/meghalaya_terrain_features_90m.nc4) | NetCDF-4 containing elevation, slope, aspect, curvature, and masks (137.9 MB). |
| **0.1° Weather Grid Terrain** | [`data/processed/terrain/meghalaya_terrain_aligned_weather_grid.nc4`](file:///d:/Downloads/honours/data/processed/terrain/meghalaya_terrain_aligned_weather_grid.nc4) | 0.1-degree summary terrain morphometry matching IMERG/MERGIR (38 KB). |
| **Antecedent Rainfall NetCDF** | [`data/processed/landslide/antecedent_rainfall_july2025.nc4`](file:///d:/Downloads/honours/data/processed/landslide/antecedent_rainfall_july2025.nc4) | Multi-timescale rainfall accumulations for 336 time intervals (2.16 MB). |
| **Landslide Events CSV** | [`outputs/landslide/meghalaya_landslide_events.csv`](file:///d:/Downloads/honours/outputs/landslide/meghalaya_landslide_events.csv) | 67 verified NASA GLC historical events with dates and coordinates (44 KB). |
| **Landslide Inventory Audit** | [`outputs/landslide/landslide_inventory_audit.json`](file:///d:/Downloads/honours/outputs/landslide/landslide_inventory_audit.json) | Detailed audit with provenance, triggers, accuracies, and caveats (23 KB). |
| **Terrain & Events Figure** | [`outputs/landslide/meghalaya_terrain_and_events.png`](file:///d:/Downloads/honours/outputs/landslide/meghalaya_terrain_and_events.png) | Publication-grade 6-panel evaluation figure (2.27 MB). |
| **Pipeline Metadata JSON** | [`outputs/landslide/terrain_and_landslide_report.json`](file:///d:/Downloads/honours/outputs/landslide/terrain_and_landslide_report.json) | Machine-readable pipeline execution summary (1.38 KB). |
