import os
from pathlib import Path

def list_summary(p):
    path = Path(p)
    if not path.exists():
        print(f"{p}: DOES NOT EXIST")
        return
    files = list(path.glob("*"))
    print(f"{p}: {len(files)} items")
    for f in sorted(files, key=lambda x: x.name)[:15]:
        size_str = f"{f.stat().st_size:,} bytes" if f.is_file() else "dir"
        print(f"  {f.name} ({size_str})")
    if len(files) > 15:
        print(f"  ... and {len(files) - 15} more")

if __name__ == "__main__":
    targets = [
        "data/raw",
        "data/imerg",
        "data/mergir",
        "data/processed/weather",
        "scripts",
        "src/models",
        "src/models/experimental",
        "outputs/models/phase3",
        "outputs/experiments",
        "outputs/experiments/model_research",
        "outputs/reports",
        "outputs/reports/model_research"
    ]
    for target in targets:
        list_summary(target)
