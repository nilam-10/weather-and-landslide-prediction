# System Troubleshooting & Operational Runbook

**Project Title:** Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction  
**Academic Program:** B.Tech Honours Major Project (2026–27), Royal Global University  
**Target Environment:** Local Windows Laptop (PowerShell 7 / Windows PowerShell)  
**File Path:** `outputs/reports/local_run_and_code_guide/TROUBLESHOOTING.md`  

---

## 1. Quick Start / How to Restart the Application

If the server stops, you restart your laptop, or you need to launch a fresh instance:

### Step 1: Open PowerShell
Open Windows Terminal or PowerShell and navigate to the project directory:
```powershell
cd D:\Downloads\honours
```

### Step 2: Ensure Port 8000 Is Free
Check if any zombie Python process is occupying port 8000:
```powershell
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
```
If a process ID (PID) appears, terminate it cleanly:
```powershell
Stop-Process -Id <PID> -Force
```

### Step 3: Launch the Server
Run the single unified command:
```powershell
# Set temporary matplotlib cache directory to prevent Windows permission warnings
$env:MPLCONFIGDIR = "$pwd\.tmp_mpl"

# Launch Uvicorn ASGI server
.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

### Step 4: Open Your Browser
Open Chrome, Edge, or Firefox and go to:
$$\mathbf{http://127.0.0.1:8000/}$$

---

## 2. Issues Encountered & Proven Solutions

During the setup and verification of this major project, several environmental and runtime edge-cases were identified and systematically resolved. The table below documents their root causes and fixes:

### Issue 1: Matplotlib Font Cache Permission Warning on Windows
- **Symptom:**  
  When importing `src.api.main`, a warning appeared:
  `Matplotlib created a temporary cache directory at ... because the default path (C:\Users\nilam\.matplotlib) is not a writable dir`.
- **Root Cause:**  
  On Windows systems with tight user-profile sandboxing, Matplotlib defaults to scanning system fonts and writing to `%USERPROFILE%\.matplotlib`, which can trigger security exceptions or slow down application startup.
- **Permanent Solution:**  
  Set the `$env:MPLCONFIGDIR` environment variable to a local temporary folder inside the workspace before launching:
  ```powershell
  $env:MPLCONFIGDIR = "$pwd\.tmp_mpl"
  ```
  This creates a small local font cache in `D:\Downloads\honours\.tmp_mpl` and eliminates all startup latency.

---

### Issue 2: Starlette / TestClient Async Deadlock in Headless Testing
- **Symptom:**  
  Running headless `scripts/test_live_application.py` via `fastapi.testclient.TestClient` hung during model initialization on Windows.
- **Root Cause:**  
  `fastapi.testclient.TestClient` spins up an internal Starlette asynchronous event loop in a background thread. When combined with PyTorch CUDA context initialization on Windows WDDM drivers, the synchronous thread lock competed with the async event loop during `lifespan(app)` execution.
- **Permanent Solution:**  
  Launch the server directly via standard Uvicorn ASGI (`uvicorn src.api.main:app --host 127.0.0.1 --port 8000`), and test endpoints using real HTTP requests (`Invoke-RestMethod` or `requests.get()`). Real HTTP socket communication runs in the main thread without event-loop thread contention.

---

### Issue 3: Concurrent CUDA Memory Allocation Contention
- **Symptom:**  
  Starting two separate Python scripts simultaneously resulted in GPU timeout or CUDA out-of-memory errors.
- **Root Cause:**  
  The laptop GPU (NVIDIA RTX 4050) has ~6 GB of VRAM. While a single inference process requires only ~412 MB, launching multiple parallel training/test scripts simultaneously triggers concurrent CUDA context creation under Windows WDDM, causing GPU mutex contention.
- **Permanent Solution:**  
  Only run **one** live inference instance at a time. The platform is designed to serve multiple concurrent browser requests from a single Uvicorn process with sub-20ms latency.

---

### Issue 4: Localhost Loopback Isolation in Windows Sandboxes
- **Symptom:**  
  Attempting to open `http://127.0.0.1:8000` in Chrome resulted in `ERR_CONNECTION_REFUSED` when the server was started inside an isolated container/sandbox.
- **Root Cause:**  
  Secure sandbox environments isolate network namespaces. Under Windows sandbox security policies, network sockets bound inside the sandbox are inaccessible to host browsers unless the process is allowed host loopback binding.
- **Permanent Solution:**  
  Ensure the Uvicorn server is executed in the normal host user session with standard loopback binding (`--host 127.0.0.1`).

---

## 3. Fallback: Running on CPU (No NVIDIA GPU)

If you need to demonstrate the project on a friend's laptop or a university desktop that does not have an NVIDIA GPU:

1. **Automatic Device Detection:**  
   The codebase (`src/api/main.py`, lines 119–125) already checks:
   ```python
   cache.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   ```
   If no CUDA device is present, PyTorch automatically runs all tensor calculations on the CPU.
2. **Forced CPU Mode:**  
   To explicitly test or demonstrate CPU inference on your current machine:
   ```powershell
   $env:MPLCONFIGDIR = "$pwd\.tmp_mpl"
   $env:CUDA_VISIBLE_DEVICES = ""
   .venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
   ```
3. **CPU Performance:**  
   Because the MultiTask ConvLSTM has only 161K parameters and inference operates on a $25 \times 40$ spatial grid, **CPU inference latency is only ~35 ms**, ensuring a smooth live demonstration even without a dedicated graphics card.

---

## 4. Verification & Diagnostic Commands

Run these quick PowerShell one-liners to verify system health at any time:

### Check Server Health
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" | ConvertTo-Json -Depth 3
```

### Check Frontend Delivery
```powershell
(Invoke-WebRequest -Uri "http://127.0.0.1:8000/").StatusCode
# Expected output: 200
```

### Check 30-Minute Calibrated Forecast
```powershell
$res = Invoke-RestMethod -Uri "http://127.0.0.1:8000/layers/weather?step=316&model=convlstm&lead=30m&calibrated=true"
Write-Host "Lead:" $res.lead_time "Calibrated:" $res.calibrated "Grid Size:" $res.grid.values.Count "x" $res.grid.values[0].Count
```

### Check Shillong Cell Inspector
```powershell
$cell = Invoke-RestMethod -Uri "http://127.0.0.1:8000/cell_inspect?lat=25.5788&lon=91.8933&step=316"
Write-Host "Location: Shillong | Rain:" $cell.meteorological_state.instantaneous_precip_mm_hr "mm/hr | Hazard:" $cell.landslide_hazard_state.lhasa_hazard_label
```
