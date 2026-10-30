# Frontend Input, UI Control & API Specification Guide

**Project:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Component:** Web GIS Frontend (`frontend/index.html`) & FastAPI Backend (`src/api/main.py`)  
**Location:** `outputs/experiments/landslide_model_comparison/FRONTEND_INPUT_AND_API_GUIDE.md`  
**Date:** October 30, 2026  

---

## 1. Complete Frontend UI Screen & Control Audit

The single-page Web GIS application (`frontend/index.html`) contains **7 functional screen panels and control clusters**. Below is the exhaustive audit of every interactive element:

### Cluster 1: Navigation Header Controls
1. **GPU Status Badge (`id="gpu-status"`):**
   - **Label:** `GPU: DETECTING...` $\to$ `GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6140 MB)`
   - **Purpose:** Informs the user whether PyTorch neural inference is hardware-accelerated or running on CPU.
   - **User Input Required:** No (auto-triggered on `window.onload`).
   - **Backend Route:** `GET /health` $\to$ `get_health()`.
   - **Behavior on Failure:** Displays `API Offline` in grey; application continues in degraded view.
2. **Scientific Provenance Button:**
   - **Label:** `📖 Provenance`
   - **Purpose:** Opens raw dataset attribution and DOI registry.
   - **User Input Required:** Click button.
   - **Backend Route:** `GET /provenance` $\to$ `get_provenance()`. Opens raw JSON manifest in new browser tab.
3. **Interactive API Documentation Button:**
   - **Label:** `⚡ API Docs`
   - **Purpose:** Opens FastAPI's auto-generated interactive OpenAPI / Swagger UI.
   - **Backend Route:** `GET /docs`. Opens in new browser tab.

---

### Cluster 2: Archive Timeline Player (Bottom Center of Map)
4. **Timeline Slider (`id="timeline-slider"`):**
   - **Label:** `ARCHIVE TIMELINE (30-MIN STEP):`
   - **Purpose:** Selects the historical observation timestamp $t$ from the July 1–7, 2025 pilot archive.
   - **User Input Required:** Yes (user drags slider or clicks timeline).
   - **Default Value:** `316` (Peak convective monsoon storm: July 7, 2025, 14:00 UTC).
   - **Data Type & Valid Range:** Integer from `0` to `335` (336 half-hourly timesteps).
   - **Sent to Backend:** Yes, as query parameter `step=316`.
   - **Endpoints Receiving It:**
     - `GET /layers/weather?step={step}`
     - `GET /layers/landslide/dynamic?step={step}`
     - `GET /layers/colocation?step={step}`
     - `GET /cell_inspect?step={step}`
   - **Screen Action:** Updates timestamp display (`#timeline-time-display`) and triggers canvas repaint for all active weather and dynamic hazard layers.
5. **Timeline Control Buttons (`#btn-prev`, `#btn-play`, `#btn-next`, `#btn-peak`):**
   - **Previous (`#btn-prev`):** Decrements step by 1 (clamped at 0).
   - **Next (`#btn-next`):** Increments step by 1 (clamped at 335).
   - **Play/Pause (`#btn-play`):** Starts a JavaScript `setInterval` advancing the step every $1,200\text{ ms}$, animating cloud advection across Meghalaya.
   - **Peak Storm (`#btn-peak`):** Instant jump button setting slider to step 316.

---

### Cluster 3: Weather Nowcasting Controls (Sidebar)
6. **Nowcasting Model Selector (`id="model-select"`):**
   - **Label:** `Nowcasting Model Candidate:`
   - **Purpose:** Chooses which neural or kinematic model generates the precipitation nowcast.
   - **Default Value:** `convlstm` (Canonical MultiTask ConvLSTM).
   - **Options:** `convlstm`, `mlp`, `fclstm`, `logistic_regression`, `optical_flow_dense`, `optical_flow_sparse`, `persistence`.
   - **Sent to Backend:** Yes, query parameter `model={value}` to `GET /layers/weather`.
7. **Forecast Horizon Selector (`id="lead-select"`):**
   - **Label:** `Forecast Horizon / Lead Time:`
   - **Purpose:** Chooses the forward prediction horizon.
   - **Default Value:** `30m` (+30 minutes, Lead 1).
   - **Options:** `30m`, `60m`, `120m` (kinematic limit), `180m`, `360m` (physical dissipation limit).
   - **Sent to Backend:** Yes, query parameter `lead={value}` to `GET /layers/weather` and `GET /explain/weather`.
8. **Platt Probability Calibration Checkbox (`id="chk-calibrate"`):**
   - **Label:** `Platt Probability Calibration:`
   - **Purpose:** Toggles between raw neural network logits and post-hoc Platt scaled probabilities.
   - **Default Value:** Checked (`true`).
   - **Sent to Backend:** Yes, query parameter `calibrated=true|false` to `GET /layers/weather`.

---

### Cluster 4: Landslide Susceptibility Controls (Sidebar)
9. **Landslide Susceptibility Model Selector (`id="landslide-model-select"`):**
   - **Label:** `Landslide Susceptibility Model (90m LSI):`
   - **Purpose:** Selects which machine-learning algorithm generates the static 90m terrain susceptibility map.
   - **Default Value:** `xgboost` (XGBoost 10-feature morphometry classifier).
   - **Options:** `xgboost`, `random_forest`, `logistic_regression`.
   - **Sent to Backend:** Yes, query parameter `model={value}` to `GET /layers/landslide/susceptibility`.
   - **Screen Action:** Replaces the static terrain heatmap with the newly selected model's output raster.

---

### Cluster 5: Independent Layer Toggles (Sidebar)
10. **Layer Toggles (`#chk-weather`, `#chk-susceptibility`, `#chk-dynamic`, `#chk-events`, `#chk-boundary`, `#chk-colocation`, `#chk-xai`):**
    - Clicking any toggle adds or removes the corresponding Leaflet `L.layerGroup` from the map canvas and updates the dynamic legend (`#legend-box`).
    - **Convective Weather Nowcast:** Renders $25 \times 40$ nowcast probability cells.
    - **Static Susceptibility (90m LSI):** Renders $173 \times 465$ terrain susceptibility raster.
    - **Dynamic Rainfall Hazard:** Renders LHASA 2.0 categorical tiers (0=Low, 1=Moderate, 2=High, 3=Very High).
    - **Verified Historical Landslides:** Renders 38 GSI/NASA GLC event markers.
    - **Meghalaya Boundary:** Renders cyan dashed polygon.
    - **Boolean Co-Location:** Renders yellow hatched overlap cells with Anti-Fallacy notice.
    - **XAI Attribution Heatmap:** Renders diverging blue-white-red attribution values.

---

### Cluster 6: Explainable AI (XAI) Nowcast Engine (Sidebar)
11. **Target Coordinate Inputs (`#xai-lat`, `#xai-lon`):**
    - **Purpose:** Sets the storm center cell $(y_c, x_c)$ whose output prediction is explained.
    - **Defaults:** Latitude `25.050°N`, Longitude `91.850°E` (Sohra escarpment).
    - **Population:** Automatically populated by clicking anywhere on the Leaflet map, or entered manually.
12. **XAI Method Selector (`#xai-method`):**
    - **Options:** `integrated_gradients` (Sundararajan et al., 2017 path integral), `occlusion` (3x3 spatial perturbation).
13. **Target Output Head (`#xai-target-type`):**
    - **Options:** `probability` (binary convective event head), `qpf` (continuous rain rate head in mm/hr).
14. **Reference Baseline Selector (`#xai-baseline`):**
    - **Options:** `dry_clearsky` (zero rain, neutral cloud Tb), `zero` (all zero normalized), `climatological` (archive mean).
15. **Compute Attribution Button (`#btn-run-xai`):**
    - **Purpose:** Executes PyTorch autograd Integrated Gradients on the RTX 4050 GPU.
    - **Screen Output:** Populates channel attribution progress bars, lookback frame temporal decay weights ($t-90$m to $t$), narrative explanation, and renders diverging attribution heatmap on the map.

---

### Cluster 7: Interactive Coordinate Cell Inspector (Sidebar)
16. **Interactive Map Click (`map.on('click')`):**
    - **Purpose:** Allows the user to inspect any coordinate in Meghalaya.
    - **Input Mechanism:** Map click captures event coordinates `[e.latlng.lat, e.latlng.lng]`.
    - **Backend Route:** `GET /cell_inspect?lat={lat}&lon={lon}&step={step}`.
    - **Returned Display:**
      - Instantaneous rain rate ($\text{mm/hr}$)
      - Cloud-top infrared brightness temperature ($\text{K}$)
      - Multi-lead nowcast probabilities (+30m, +60m)
      - Static Landslide Susceptibility Index (LSI, 0.000 to 1.000) and tier badge
      - Cumulative Antecedent Rainfall (mm)
      - NASA LHASA 2.0 Dynamic Hazard level and advisory tier

---

## 2. Complete REST API Inventory

### Endpoint 1: System Health & Hardware Diagnostics
- **Method & Path:** `GET /health`
- **Purpose:** Detects active GPU accelerator, VRAM allocation, and dataset readiness.
- **Query Parameters:** None.
- **Request Headers:** Standard HTTP (`Accept: application/json`).
- **Authentication:** None (Public local endpoint).
- **Example Request:**
  ```http
  GET http://127.0.0.1:8000/health HTTP/1.1
  ```
- **Example Response:**
  ```json
  {
    "status": "online",
    "service": "Multimodal Convective Nowcasting & Landslide Risk API",
    "version": "1.1.0",
    "hardware": {
      "cuda_available": true,
      "device": "cuda:0",
      "gpu_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
      "vram_total_mb": 6140.5
    },
    "datasets": {
      "weather_steps_available": 336,
      "historical_landslide_events_verified": 38
    }
  }
  ```

---

### Endpoint 2: Weather Nowcast Inference
- **Method & Path:** `GET /layers/weather`
- **Purpose:** Runs real-time forward inference through the selected neural or kinematic model.
- **Query Parameters:**
  - `step` (int, default=316, range=0..335): Target historical timestep.
  - `model` (str, default="convlstm"): Model identifier (`convlstm`, `mlp`, `fclstm`, `logistic_regression`, `optical_flow_dense`, `optical_flow_sparse`, `persistence`).
  - `lead` (str, default="30m"): Forecast horizon (`30m`, `60m`, `120m`, `180m`, `360m`).
  - `calibrated` (bool, default=true): Enables Platt probability calibration.
- **Example Request:**
  ```http
  GET http://127.0.0.1:8000/layers/weather?step=316&model=convlstm&lead=30m&calibrated=true HTTP/1.1
  ```
- **Example Response:**
  ```json
  {
    "layer_type": "convective_weather_nowcast",
    "reference_time_utc": "2025-07-07T14:00:00Z",
    "lead_time": "30m",
    "model_name": "convlstm",
    "calibrated": true,
    "grid": {
      "latitudes": [24.55, 24.65, "... 25 values ..."],
      "longitudes": [89.55, 89.65, "... 40 values ..."],
      "values": [[0.01, 0.05, 0.82, "... 25x40 array ..."]],
      "rain_rate_mm_hr": [[0.0, 0.2, 18.4, "... 25x40 array ..."]]
    }
  }
  ```

---

### Endpoint 3: Static Landslide Susceptibility (90m LSI)
- **Method & Path:** `GET /layers/landslide/susceptibility`
- **Purpose:** Serves high-resolution terrain susceptibility index derived from 90m SRTM DEM morphometry.
- **Query Parameters:**
  - `model` (str, default="xgboost"): `xgboost`, `random_forest`, `logistic_regression`.
- **Example Request:**
  ```http
  GET http://127.0.0.1:8000/layers/landslide/susceptibility?model=xgboost HTTP/1.1
  ```
- **Example Response:**
  ```json
  {
    "layer_type": "landslide_static_susceptibility",
    "model_name": "xgboost",
    "resolution_m": 720.0,
    "spatial_cross_val_roc_auc": 0.5913,
    "grid": {
      "latitudes": [26.15, "... 173 values ..."],
      "longitudes": [89.75, "... 465 values ..."],
      "values": [[null, 0.21, 0.74, "... 173x465 array ..."]]
    }
  }
  ```

---

### Endpoint 4: Dynamic Antecedent Rainfall Hazard (NASA LHASA 2.0)
- **Method & Path:** `GET /layers/landslide/dynamic`
- **Purpose:** Computes rolling Antecedent Rainfall Index ($A = 0.6 P_{24\text{h}} + 0.4 P_{72\text{h}}$) and assigns situational awareness hazard levels (0–3).
- **Query Parameters:**
  - `step` (int, default=316, range=0..335): Target archive timestep.
- **Example Request:**
  ```http
  GET http://127.0.0.1:8000/layers/landslide/dynamic?step=316 HTTP/1.1
  ```
- **Example Response:**
  ```json
  {
    "layer_type": "landslide_dynamic_hazard",
    "step": 316,
    "formula": "ARI = 0.6 * P24h + 0.4 * P72h",
    "category_counts": {
      "Level_0_Low": 978,
      "Level_1_Moderate": 18,
      "Level_2_High": 4,
      "Level_3_VeryHigh": 0
    },
    "grid": {
      "hazard_levels": [[0, 0, 1, 2, "... 25x40 integer array ..."]],
      "antecedent_rain_mm": [[12.4, 45.1, 112.8, "... 25x40 float array ..."]]
    }
  }
  ```

---

### Endpoint 5: Verified Historical Landslide Catalog
- **Method & Path:** `GET /layers/landslide/events`
- **Purpose:** Delivers verified GeoJSON FeatureCollection of historical rainfall-induced landslides ($N=38$).
- **Query Parameters:** None.
- **Example Response:**
  ```json
  {
    "type": "FeatureCollection",
    "total_features": 38,
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [91.8933, 25.5788]
        },
        "properties": {
          "title": "Fatal landslide in Umiam",
          "date": "2017-06-17",
          "trigger": "continuous_rain",
          "accuracy": "5km",
          "spatial_block": "Central"
        }
      }
    ]
  }
  ```

---

### Endpoint 6: Dual-Stack Coordinate Cell Inspector
- **Method & Path:** `GET /cell_inspect`
- **Purpose:** Ingests latitude and longitude and returns both weather state and landslide susceptibility.
- **Query Parameters:**
  - `lat` (float, required, $24.5 \le \text{lat} \le 27.0$)
  - `lon` (float, required, $89.5 \le \text{lon} \le 93.5$)
  - `step` (int, default=316)
- **Example Request:**
  ```http
  GET http://127.0.0.1:8000/cell_inspect?lat=25.2700&lon=91.7300&step=316 HTTP/1.1
  ```
- **Example Response:**
  ```json
  {
    "query_coordinates": {"lat": 25.27, "lon": 91.73},
    "matched_weather_cell": {"center_lat": 25.25, "center_lon": 25.75},
    "meteorological_state": {
      "instantaneous_precip_mm_hr": 18.4,
      "infrared_brightness_temp_k": 218.4,
      "nowcast_probabilities": {
        "lead_30min": 0.82,
        "lead_60min": 0.74,
        "lead_120min": 0.51,
        "lead_180min": 0.32,
        "lead_360min": 0.08
      }
    },
    "landslide_hazard_state": {
      "static_susceptibility_index": 0.742,
      "susceptibility_tier": "High",
      "antecedent_rainfall_trigger_mm": 112.4,
      "lhasa_dynamic_hazard_level": 2,
      "lhasa_hazard_label": "High"
    }
  }
  ```

---

### Endpoint 7: Explainable AI (Integrated Gradients)
- **Method & Path:** `GET /explain/weather`
- **Purpose:** Executes path integral gradient attribution to quantify historical input feature contributions.
- **Query Parameters:** `step`, `lead`, `lat`, `lon`, `method`, `target_type`, `baseline`.
- **Response Structure:** Includes completeness proof (`axiom_passed`), channel shares (`precipitation_pct`, `infrared_tb_pct`), temporal lookback weights, and 2D spatial attribution matrix.

---

## 3. How the Frontend Obtains Model Inputs

A critical architectural design of this platform is that **the user is never required to manually type raw geomorphometric parameters (such as slope degrees, profile curvature, or rainfall millimeters)**:
1. **Coordinate Derivation:** When the user clicks the map, Leaflet emits geographical coordinates $(\text{lat}, \text{lon})$.
2. **Backend Spatial Lookup:** The backend locates the nearest grid cell in the preprocessed 90m SRTM terrain NetCDF (`meghalaya_terrain_features_90m.nc4`) and the weather cube NetCDF (`aligned_weather_mergir_YYYYMMDD.nc4`).
3. **Automated Feature Ingestion:** The 8 terrain features and antecedent rainfall values are retrieved directly from the memory cache and passed into the model pipelines in sub-5ms latency.
4. **Conclusion:** Selecting a location on the interactive map is **$100\%$ sufficient** to execute real-time dual-stack predictions.
