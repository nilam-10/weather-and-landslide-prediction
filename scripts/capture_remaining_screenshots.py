"""Capture Remaining Browser Screenshots for Phase 5."""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "outputs/.matplotlib"
FIG_DIR = Path("outputs/figures/phase5")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not Path(CHROME).exists():
    CHROME = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

p = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000"])
try:
    print("Waiting for server...")
    for _ in range(40):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1) as resp:
                if resp.status == 200:
                    print("Server ready!")
                    break
        except Exception:
            time.sleep(0.5)

    # Fig 03: Mobile Responsive Layout
    out3 = FIG_DIR / "fig03_frontend_mobile_layout.png"
    print(f"Capturing {out3}...")
    cmd3 = [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--hide-scrollbars",
        "--window-size=414,896",
        f"--screenshot={out3.resolve()}",
        "http://127.0.0.1:8000/",
    ]
    subprocess.run(cmd3, timeout=15)
    print(f"  {out3.name}: exists={out3.exists()}, size={out3.stat().st_size if out3.exists() else 0}")

    # Fig 04: API Documentation
    out4 = FIG_DIR / "fig04_frontend_fastapi_docs.png"
    print(f"Capturing {out4}...")
    cmd4 = [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--hide-scrollbars",
        "--window-size=1280,800",
        f"--screenshot={out4.resolve()}",
        "http://127.0.0.1:8000/docs",
    ]
    subprocess.run(cmd4, timeout=15)
    print(f"  {out4.name}: exists={out4.exists()}, size={out4.stat().st_size if out4.exists() else 0}")

finally:
    p.terminate()
    p.wait()
    print("Server cleanly shut down.")
