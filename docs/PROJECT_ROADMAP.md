# Honours hazard project: staged data and training roadmap

## Confirmed sources and scientific scope

The uploaded RBU project synopsis identifies Meghalaya as the initial study area. The two modules are (A) satellite/rainfall-based short-term weather hazard probabilities and (B) dated-inventory/rainfall/terrain-based next-day landslide probabilities. The result is a GIS frontend, with separate layers, not a third joint probability model.

### Priority download table

| Priority | Dataset | Product / format | First acquisition | Full-project acquisition rule | Primary use |
|---|---|---|---|---|---|
| 0 | Existing sample | `3RIMG_L1C_ASIA_MER` HDF5, six frames from Feb 1, 2025 | already available | no re-download | inspect, geolocate, convert LUT |
| 1 | Meghalaya polygon | authoritative/licensable vector boundary GeoJSON or Shapefile | one-time | define a documented bounding buffer | exact AOI masking and common grid |
| 1 | GPM IMERG | NASA GES DISC `GPM_3IMERGHH_07` version 07B (Final, half-hourly) and, if latency matters, Early `GPM_3IMERGHHE_07` | Feb 1, 2025 same UTC sample (smoke test) | dates matching INSAT and credible landslide-event dates; spatial subset | weather proxy target and rain histories |
| 1 | GSI / NRSC events | coordinates, date/time precision, event provenance; GeoJSON, Shapefile or CSV when licensed | request/export Meghalaya dated events + metadata | choose model year(s) AFTER event coverage gate | landslide supervised labels |
| 1 | SRTM DEM | USGS 1 arc-second global GeoTIFF (~30m) | Meghalaya area, once | same static DEM reused | terrain features |
| 2 | INSAT monsoon seasons | MOSDAC `3RIMG_L1C_ASIA_MER` HDF5; may be whole Asia before local cropping | one small event-matched period, e.g. 2–4 weeks | at least 2–3 compatible rainy seasons if inventory supports | nowcasting deep model |
| 3 | DWR / lightning | actual authorized, dated records | verify access ONLY | if continuity permits | optional supervised labels/modalities |
| 3 | Optional terrain factors | land-cover/soil-moisture | after core models | with comparable metadata | optional ablation |

### Exact official entry points

- MOSDAC dataset description: https://mosdac.gov.in/doi/164/
- MOSDAC download API user manual: https://mosdac.gov.in/downloadapi-manual
- MOSDAC API package: https://www.mosdac.gov.in/software/mdapi.zip
- MOSDAC catalogue for datasetId: https://mosdac.gov.in/catalog-app/satellite.php
- NASA Final half-hourly: https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_07/summary
- NASA Early half-hourly: https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHHE_07/summary
- NASA Earthdata account: https://urs.earthdata.nasa.gov/
- USGS EarthExplorer: https://earthexplorer.usgs.gov/
- GSI landslides: https://gsi.gov.in/landslide-hazard/
- NRSC atlas: https://www.nrsc.gov.in/nrscnew/resources_atlas_landslide.php

**Access caveat:** NRSC atlas pages are not guaranteed to contain an exportable, dated, Meghalaya-specific supervised training inventory; permission or request may be necessary. Research models must not use a map screenshot as if it were a validated label table.

## Critical corrections to earlier assumptions

- The 132MiB ZIP is six frames for **one day**, not 90 days. File `error.log` states `downloader::recursive Timeout`.
- A weather precipitation target at 15-minute resolution is **not** validated from 30-minute IMERG, so start with 30/60/180 minute leads.
- Without independent radar/lightning/hail labels, call the target satellite-rainfall threshold exceedance (proxy), not measured convective thunderstorm occurrence.
- IMERG Final is **retrospective research-grade** and not available immediately; use Early with appropriate latency for a prospective/operationally realistic input scenario.
- Event inventories without dates can support susceptibility only, not tomorrow's event probability.
- A month of monsoon observations without a dated landslide event inventory does not establish a supervised landslide dataset.
- For SRTM ~30m terrain and IMERG ~0.1-degree rain, interpolation changes sampling density but cannot create fine-scale rainfall measurements.

## Pilot schedule

1. **Day 1:** Run `scripts/inspect_insat.py` on the existing ZIP, save inventory+gap audit; agree on precise Meghalaya polygon and time availability rules.
2. **Days 2–3:** Download IMERG for exactly Feb 1, 2025; validate units/time masks/CRS alignment against six INSAT frames. Download/define Meghalaya polygon.
3. **Days 3–5:** Obtain SRTM DEM tiles and construct elevation/slope/aspect/curvature previews. Start official requests for dated event inventory.
4. **Week 2:** Obtain small overlapping monsoon window, create common weather grid, time audit, 30min rainfall proxy label, persistence/optical flow baseline; if inventory arrives build first grid-day event labels.
5. **Weeks 3–4:** Expand only after weather/missingness checks and a confirmed event-date access gate; implement Logistic Regression and boosted trees, train/evaluate with honest splits.
6. **Weeks 5–8:** Add ConvLSTM only if enough rainy/dry examples and multiple years/seasons exist. Implement model calibration, ablations, near-operational latency simulation, dashboard.
7. **Remaining planned 24-week schedule:** deeper published-model benchmark, held-out performance, paper/report, frontend robustness and packaged reproducibility.

## Weather data budget (order of magnitude)

Each supplied full-Asia HDF5 file is around 21–26 MB. Nominal half-hourly cadence gives 48/day (~1.1 GB/day), 1440/30 days (~34 GB), 4320/90 days (~100 GB), and ~17,520/year (~400 GB), **before** data absences and product changes. A 2–3-year full-Asia archive could exceed 0.8–1.2 TB. Check exact file counts via MOSDAC API search before ordering, crop promptly to Meghalaya, archive carefully, and store processed chunked crops for Colab. The official API manual confirms a maximum of **5,000 data files per user per day**.

## Labels and holdouts

Weather: supervised threshold exceedance of subsequent IMERG precipitation at supported leads, not direct lightning/hail. Use temporal separation of overlapping sliding windows; split at season/chronological-block level before normalization.

Landslide: features only known by prediction cutoff, one row per cell/day; positive when a properly dated event occurs within the target period. Unknown event sites/time periods are not verified non-events. Stratified candidate controls, split by distinct geographic regions and time, evaluate prevalence/calibration implications. Report inability to train event prediction if dated records cannot be obtained.

## Compute split

- **Windows local VS Code:** inspect HDF5, QGIS, download sources, DEM features, dataset audits, CSV/Parquet prep, classic baselines, API/dashboard.
- **Colab CPU:** small sklearn/XGBoost baseline experiments if local CPU constrained.
- **Colab GPU:** cropped 4–8-frame weather ConvLSTM (channels and masks, lead heads), conditional landslide GRU only when justified. Save checkpoints and logs to Drive; never download all full-Asia HDFs into Colab or rely on ephemeral `/content` for durable storage.

## Four-person team split

1. Source acquisition and audit (MOSDAC/INSAT), schema and weather input availability.
2. NASA rainfall + nowcasting labels, baselines, ConvLSTM and metrics.
3. Landslide event inventory + DEM + tabular baseline / conditional GRU, calibration, SHAP.
4. Split registry / leakage audits, FastAPI + GIS integration, experiments, report and deployment. All teammates review scientific claims and share one manifest/split registry.

## Definition of successful final project

- Two validated independently trained research model paths, or transparently scoped substitutes where labels are inaccessible.
- Actual training/evaluation records, not imagined numbers.
- Baseline comparisons, ablations, calibration and leakage-safe spatial and temporal splits.
- True provenance/time/data-age in independent GIS layers, honest unavailable states and explicit 'research use only' disclaimer.
- Reproducible code plus honest dataset access/licensing notes and limitations.
