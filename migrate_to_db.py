"""
One-time migration: loads local pkl/CSV files into PostgreSQL (Neon).

Run once from the FastAPIs directory after setting DATABASE_URL:

    set DATABASE_URL=postgresql://...   (Windows)
    export DATABASE_URL=postgresql://... (Mac/Linux)
    python migrate_to_db.py

The script is safe to re-run — it uses if_exists="replace" so tables are
rebuilt from scratch each time. Do NOT run while the API is serving traffic
as it briefly drops and recreates the tables.
"""

import os
import pickle
import pandas as pd
from sqlalchemy import create_engine, text

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "")
if not DATABASE_URL:
    raise SystemExit("ERROR: Set the DATABASE_URL environment variable before running this script.")

script_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(script_dir, "models")

# data_viz1.csv lives in the sibling Streamlit Frontend folder.
# Override with VIZ_CSV_PATH env var if your folder layout differs.
VIZ_CSV_PATH = os.environ.get(
    "VIZ_CSV_PATH",
    os.path.join(script_dir, "..", "Streamlit Frontend", "data", "data_viz1.csv")
)

# ── Connect ───────────────────────────────────────────────────────────────────
print("Connecting to PostgreSQL...")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
with engine.connect() as conn:
    result = conn.execute(text("SELECT version()"))
    print(f"Connected: {result.fetchone()[0][:50]}...\n")

# ── Table 1: property_features (from input_data_X.pkl) ───────────────────────
# This powers /options and /sectors — the dropdown data for the prediction form.
print("Migrating property_features...")
pkl_path = os.path.join(MODELS_DIR, "input_data_X.pkl")
with open(pkl_path, "rb") as f:
    features_df = pickle.load(f)

print(f"  Shape: {features_df.shape}")
print(f"  Columns: {features_df.columns.tolist()}")

features_df.to_sql("property_features", engine, if_exists="replace", index=False)
print(f"  Done — {len(features_df)} rows written to property_features\n")

# ── Table 2: properties_viz (from data_viz1.csv) ─────────────────────────────
# This powers the /analytics/data endpoint and the Analytics dashboard.
print("Migrating properties_viz...")
viz_path = os.path.normpath(VIZ_CSV_PATH)
if not os.path.exists(viz_path):
    raise FileNotFoundError(
        f"data_viz1.csv not found at: {viz_path}\n"
        "Set VIZ_CSV_PATH env var to the correct absolute path."
    )

viz_df = pd.read_csv(viz_path)
print(f"  Shape: {viz_df.shape}")
print(f"  Columns: {viz_df.columns.tolist()}")

viz_df.to_sql("properties_viz", engine, if_exists="replace", index=False)
print(f"  Done — {len(viz_df)} rows written to properties_viz\n")

# ── Verify ────────────────────────────────────────────────────────────────────
print("Verifying row counts...")
with engine.connect() as conn:
    pf_count = conn.execute(text("SELECT COUNT(*) FROM property_features")).scalar()
    pv_count = conn.execute(text("SELECT COUNT(*) FROM properties_viz")).scalar()

print(f"  property_features : {pf_count} rows (expected {len(features_df)})")
print(f"  properties_viz    : {pv_count} rows (expected {len(viz_df)})")

if pf_count == len(features_df) and pv_count == len(viz_df):
    print("\nMigration complete!")
else:
    raise RuntimeError("Row count mismatch — check the output above for errors.")

engine.dispose()
