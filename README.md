# Honours: Multimodal Weather Nowcasting & Landslide Risk

This starter kit contains **an actually tested INSAT HDF5 inspector**, a detailed Codex prompt, and a data/Colab roadmap. It does **not** pretend the models are trained or that the missing IMERG, SRTM, and GSI/NRSC datasets are present.

## Start on Windows

1. Install Python 3.12, VS Code, and QGIS.
2. Create a working folder outside the ZIP, copy this starter kit into it.
3. In the project folder open a PowerShell terminal:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-inspection.txt
.\.venv\Scripts\python.exe scripts\inspect_insat.py --zip "D:\path\to\Oct26_192660.zip" --out outputs\inspection
```

Outputs: `insat_inventory.csv`, `sample_report.json`, and `tir1_meghalaya_preview.png`. `scripts/inspect_insat.py` was run successfully against the user's uploaded sample. It uses an approximate buffer rectangle strictly for visualization; **a true Meghalaya vector polygon is still required for modelling**.

## Give Codex the prompt

Open the starter folder in Codex/VS Code; place the original ZIP in a locally accessible (non-Git) `data/raw/` folder or supply its path. Open and paste `docs/CODEX_MASTER_PROMPT.md`. Tell Codex explicitly to **start with M0 only** and read `README.md` and the files first. Only when tests pass proceed to M1, then M2, etc.

## Research sources and next steps

See `docs/PROJECT_ROADMAP.md` for the source-by-source download choices, data quality caveats, computational plan and 24-week milestone strategy.

## Scientific honesty

- Example images are research previews, not forecasts.
- Final IMERG data cannot appear magically in a near-real-time model.
- Dateless landslide inventories cannot support next-day supervised targets.
- No official warning claims are made.
