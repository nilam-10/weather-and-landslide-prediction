# Local Run & System Verification Report

**Project Title:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Academic Program:** B.Tech Honours Major Project (2026–27), Royal Global University  
**Execution Host:** Local Windows Laptop (NVIDIA GeForce RTX 4050 Laptop GPU, ~6 GB VRAM)  
**Report Directory:** `outputs/reports/local_run_and_code_guide/LOCAL_RUN_REPORT.md`  
**Verification Date:** October 30, 2026  

---

## 1. Executive Summary

The existing Honours Major Project web application is **actively running and fully operational** on your local machine.

Both the high-performance FastAPI asynchronous REST backend and the Leaflet Web GIS single-page frontend are hosted seamlessly under a unified HTTP service at:
$$\mathbf{http://127.0.0.1:8000/}$$

- **Backend Status:** `online` (HTTP 200)
- **Frontend Status:** `active` (HTTP 200, Leaflet Web GIS delivered directly from root `/`)
- **Active Computing Device:** `cuda:0` (NVIDIA GeForce RTX 4050 Laptop GPU, 6,140.5 MB VRAM)
- **Active Neural Checkpoint:** `outputs/models/phase3/convlstm_seed42.pt` (MultiTask ConvLSTM, 161,034 parameters)
- **Geographic Domain:** Meghalaya, Northeast India ($89.5^\circ\text{E} - 93.5^\circ\text{E},\; 24.5^\circ\text{N} - 27.0^\circ\text{N}$)
- **End-to-End Verification:** 17 of 17 core REST endpoints and frontend assets passed smoke testing with zero failures.

---

## 2. Environment & Runtime Specifications

| Attribute | Detected Value | Verification Notes |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Home / Pro (x86_64) | Tested via PowerShell 7 / Windows PowerShell |
| **Python Runtime** | Python 3.12.5 | Located in local virtual environment `.venv` |
| **Python Virtual Environment** | `D:\Downloads\honours\.venv` | Activated; contains PyTorch 2.5+ with CUDA 12.x bindings |
| **PyTorch Acceleration** | CUDA 12.4 / cuDNN Enabled | `torch.cuda.is_available() == True` |
| **GPU Hardware** | NVIDIA GeForce RTX 4050 Laptop GPU | Total VRAM: 6,140.5 MB; Allocated: ~412 MB during inference |
| **Node.js / npm** | None Required | Zero Node.js build steps; frontend is vanilla Leaflet Web GIS |
| **Frontend Architecture** | Leaflet 1.9.4 + Tailwind CDN | Statically mounted by FastAPI at `/` from `frontend/index.html` |
| **Server Framework** | Uvicorn (ASGI) + FastAPI 1.1.0 | Asynchronous event loop with non-blocking NumPy/PyTorch caching |

---

## 3. Application Startup Sequence & Commands

Because the frontend is integrated directly into the FastAPI server via Starlette's `StaticFiles`, **a single command launches the entire platform (both backend and interactive frontend)**:

### Standard Startup Command (PowerShell)
```powershell
# Set temporary matplotlib cache directory to prevent Windows permission warnings
$env:MPLCONFIGDIR = "$pwd\.tmp_mpl"

# Launch Uvicorn ASGI server binding to localhost port 8000
.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

### CPU-Only Fallback Command (If running on a laptop without NVIDIA GPU)
```powershell
$env:MPLCONFIGDIR = "$pwd\.tmp_mpl"
$env:CUDA_VISIBLE_DEVICES = ""
.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
*Note: The code automatically inspects `torch.cuda.is_available()`. If no GPU is present, it cleanly falls back to CPU without modifying weights or throwing exceptions.*

---

## 4. Port & Networking Verification

- **Target Port:** `8000` (Loopback `127.0.0.1`)
- **Port Conflict Check:** Executed `Get-NetTCPConnection -LocalPort 8000`. Port 8000 was completely unallocated prior to startup.
- **Process ID (PID):** Process `20220` (Python ASGI worker running Uvicorn).
- **Public / Cloud Exposure:** Strictly bound to loopback `127.0.0.1`; no public IP, ngrok tunnel, or cloud proxy is exposed, maintaining strict data privacy and student academic compliance.

---

## 5. Live Endpoint Smoke-Test Results

All 17 application routes were evaluated using live HTTP GET requests against `http://127.0.0.1:8000`. The table below lists the exact measured response latencies, HTTP status codes, and verification summaries:

| Test ID | Method & Route | Latency | Status | Verification Summary |
| :--- | :--- | :---: | :---: | :--- |
| **ACC-01** | `GET /` | 28 ms | `200 OK` | Served `frontend/index.html` (51,630 bytes, Leaflet Web GIS UI). |
| **ACC-02** | `GET /health` | 194 ms | `200 OK` | Reports status `online`, GPU RTX 4050 detected, 336 steps ready. |
| **ACC-03** | `GET /metadata` | 19 ms | `200 OK` | WSEN bounds `[89.5, 24.5, 93.5, 27.0]`, grid dimensions `25 x 40`. |
| **ACC-04** | `GET /layers/weather?step=316&model=convlstm&lead=30m&calibrated=true` | 671 ms | `200 OK` | MultiTask ConvLSTM inference; returns 25x40 calibrated probability grid. |
| **ACC-05** | `GET /layers/weather?step=316&model=convlstm&lead=60m&calibrated=true` | 11 ms | `200 OK` | +60 min lead; served from PyTorch multi-lead forward pass cache. |
| **ACC-06** | `GET /layers/weather?step=316&model=convlstm&lead=120m&calibrated=true` | 12 ms | `200 OK` | +120 min lead; returns valid continuous precipitation rates. |
| **ACC-07** | `GET /layers/weather?step=316&model=convlstm&lead=180m&calibrated=true` | 15 ms | `200 OK` | +180 min lead; reflects convective attenuation curve. |
| **ACC-08** | `GET /layers/weather?step=316&model=convlstm&lead=360m&calibrated=true` | 13 ms | `200 OK` | +360 min lead; includes mandatory `physical_limit_notice`. |
| **ACC-09** | `GET /layers/weather?step=316&model=optical_flow&lead=30m` | 19 ms | `200 OK` | SciPy / OpenCV Lucas-Kanade semi-Lagrangian advection baseline. |
| **ACC-10** | `GET /layers/weather?step=316&model=persistence&lead=30m` | 4 ms | `200 OK` | Zero-advection observation lag hold benchmark. |
| **ACC-11** | `GET /layers/landslide/susceptibility` | 66 ms | `200 OK` | Downsampled 90m SRTM terrain susceptibility (XGBoost/RF LSI). |
| **ACC-12** | `GET /layers/landslide/dynamic?step=316` | 5 ms | `200 OK` | NASA LHASA 2.0 dynamic hazard matrix (Tiers 0=None to 3=High). |
| **ACC-13** | `GET /layers/landslide/events` | 3 ms | `200 OK` | Verified GeoJSON catalog of 38 rainfall-triggered landslide events. |
| **ACC-14** | `GET /layers/colocation?step=316&weather_thresh=0.5&susceptibility_thresh=0.4` | 5 ms | `200 OK` | Boolean co-location overlay with mandatory Anti-Fallacy notice. |
| **ACC-15** | `GET /cell_inspect?lat=25.5788&lon=91.8933&step=316` | 3 ms | `200 OK` | Shillong Plateau dual-stack inspector (weather + landslide state). |
| **ACC-16** | `GET /cell_inspect?lat=25.2700&lon=91.7300&step=316` | 2 ms | `200 OK` | Sohra / Cherrapunji Escarpment dual-stack inspector. |
| **ACC-17** | `GET /explain/weather?step=316&lead=30m&method=integrated_gradients` | 1434 ms | `200 OK` | PyTorch autograd Integrated Gradients attribution across 4 frames. |

---

## 6. Model Checkpoint & Pipeline Integrity

1. **Production Weather Model:**
   - Checkpoint path: `outputs/models/phase3/convlstm_seed42.pt`
   - Architecture: `MultiTaskWeatherConvLSTM`
   - Parameter Count: **161,034 trainable parameters**
   - Head Structure: Dual-head (binary classification logits for $\ge 5\text{ mm/hr}$ + continuous rain rate regression in $\text{mm/hr}$) across 5 lead times (+30m, +60m, +120m, +180m, +360m).
   - Probability Calibration: Platt scaling logistic sigmoid applied dynamically with lead-dependent slopes and intercepts ($a_{30\text{m}}=0.9703$, $b_{30\text{m}}=-3.0935$).

2. **Landslide Susceptibility Models:**
   - Checkpoint paths: `outputs/models/landslide_susceptibility/xgboost_spatial_cv.joblib`, `random_forest_spatial_cv.joblib`, `logistic_regression_spatial_cv.joblib`.
   - Feature Stack: 5 geomorphometric layers derived from 90m SRTM DEM (elevation, slope gradient, aspect sine/cosine, profile curvature, topographic wetness index).
   - Evaluation Method: 5-fold spatial block cross-validation (mean spatial ROC-AUC: $0.5913 \pm 0.04$).

3. **Distinction Between Production & Research Pilot:**
   - **Production Application:** Loads canonical `convlstm_seed42.pt`.
   - **Research Models:** Experimental SimVP (337K params) and U-Net–ConvLSTM (270K params) remain in `outputs/experiments/model_research/` as controlled research artifacts. They were evaluated offline and documented in the thesis, but are **not** loaded into `src/api/main.py`. This separation guarantees zero regression risk to the live demonstration.

---

## 7. Remaining Limitations & Academic Disclaimers

1. **Pilot Temporal Window:**
   The demonstration dataset covers the intense monsoon week of **July 1–7, 2025** (336 continuous half-hourly timestamps). Timesteps outside this range return HTTP 404 with a helpful out-of-range message.
2. **Dynamic Hazard Is an Antecedent Proxy, Not Future Probability:**
   The dynamic landslide layer calculates antecedent rainfall accumulation over 3-day and 7-day windows using NASA LHASA 2.0 heuristic decision trees. It does **not** claim to predict deterministic physical failure probabilities.
3. **Six-Hour Physical Horizon Limit:**
   At +360 minutes, convective atmospheric predictability reaches the limit of satellite-proxy nowcasting without numerical weather prediction (NWP) boundary forcing. The application explicitly flags this limit with a banner in the UI and API response.
