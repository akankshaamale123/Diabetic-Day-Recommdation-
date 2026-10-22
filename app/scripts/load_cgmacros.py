# app/scripts/load_cgmacros.py
from pathlib import Path
from storage import cgmacros_loader
from storage import db

# Looks for: app/data/raw/CGMacros
ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "CGMacros"

if __name__ == "__main__":
    if not ROOT.is_dir():
        raise SystemExit(f"CGMacros folder not found: {ROOT}")

    print(f"Loading from: {ROOT}\n")
    
    # Optional: Clear old empty database if it exists (Start fresh)
    db.init_db()
    db.clear_all()
    
    results = cgmacros_loader.import_all(ROOT)
    
    total_g = sum(r["glucose_rows"] for r in results)
    total_m = sum(r["meal_rows"] for r in results)
    
    print(f"\n✅ Loaded {len(results)} subjects · {total_g} glucose readings · {total_m} meal rows")