"""
One-time script: uploads ML artifact pkl files to Cloudflare R2.

Run from the FastAPIs directory with the four R2 env vars set:

    $env:R2_ENDPOINT_URL      = "https://<accountid>.r2.cloudflarestorage.com"
    $env:R2_ACCESS_KEY_ID     = "<your-access-key-id>"
    $env:R2_SECRET_ACCESS_KEY = "<your-secret-access-key>"
    $env:R2_BUCKET            = "real-estate-models"
    python upload_models_to_r2.py

Safe to re-run — it overwrites existing objects in the bucket.
input_data_X.pkl is intentionally excluded: it is now served from PostgreSQL.
"""

import os
import sys
import boto3
from pathlib import Path

# ── Validate env vars ─────────────────────────────────────────────────────────
required = ["R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"]
missing = [k for k in required if not os.environ.get(k)]
if missing:
    sys.exit(f"ERROR: Missing env vars: {', '.join(missing)}\nSee script docstring for usage.")

R2_ENDPOINT_URL      = os.environ["R2_ENDPOINT_URL"]
R2_ACCESS_KEY_ID     = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET            = os.environ["R2_BUCKET"]

# ── Files to upload ───────────────────────────────────────────────────────────
# These are the ML artifacts that live in the Docker image today.
# After uploading, the backend will download them from R2 at startup instead.
MODELS_DIR = Path(__file__).parent / "models"
ARTIFACTS = [
    "final_xgb_pipeline.pkl",   # XGBoost price prediction pipeline  (4.7 MB)
    "cosine_sim1.pkl",           # Recommender similarity matrix 1    (473 KB)
    "cosine_sim2.pkl",           # Recommender similarity matrix 2    (473 KB)
    "cosine_sim3.pkl",           # Recommender similarity matrix 3    (473 KB)
    "location_distance.pkl",     # Location distance matrix           (199 KB)
]

# ── Connect ───────────────────────────────────────────────────────────────────
print(f"Connecting to R2 bucket '{R2_BUCKET}'...")
s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
)

# ── Upload ────────────────────────────────────────────────────────────────────
print()
for filename in ARTIFACTS:
    path = MODELS_DIR / filename
    if not path.exists():
        sys.exit(f"ERROR: File not found: {path}")
    size_kb = path.stat().st_size / 1024
    print(f"  Uploading {filename} ({size_kb:.0f} KB)...", end=" ", flush=True)
    s3.upload_file(str(path), R2_BUCKET, filename)
    print("done")

# ── Verify ────────────────────────────────────────────────────────────────────
print("\nVerifying uploads in bucket...")
response = s3.list_objects_v2(Bucket=R2_BUCKET)
objects = {obj["Key"]: obj["Size"] for obj in response.get("Contents", [])}

all_ok = True
for filename in ARTIFACTS:
    if filename in objects:
        print(f"  OK  {filename}  ({objects[filename] / 1024:.0f} KB in bucket)")
    else:
        print(f"  MISSING  {filename}")
        all_ok = False

if not all_ok:
    sys.exit("\nERROR: Some files are missing from the bucket — check the output above.")

print(f"\nAll {len(ARTIFACTS)} artifacts uploaded successfully to '{R2_BUCKET}'.")
print("\nNext step: set the four R2_* env vars on your Render backend service.")
print("The backend will download from R2 on next startup instead of reading local files.")
