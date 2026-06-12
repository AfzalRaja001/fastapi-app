import io
import os
import json
import pickle
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Config 
base_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(base_dir, "models"))
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")

# Auth: leave empty to keep current open behaviour; set in Render env to enforce.
API_KEY = os.environ.get("API_KEY", "")

# Database: leave empty to load from local pkl files (current behaviour).
# Set to a Neon/Postgres connection string to serve data from the DB instead.
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# R2 object storage: leave all empty to load artifacts from local models/ dir.
# Set all four to download pkl artifacts from Cloudflare R2 at startup instead.
R2_ENDPOINT_URL     = os.environ.get("R2_ENDPOINT_URL", "")
R2_ACCESS_KEY_ID    = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET           = os.environ.get("R2_BUCKET", "")

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# API-key auth 
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def require_api_key(key: str = Depends(_api_key_header)):
    """Enforce X-API-Key header only when API_KEY env var is set."""
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# Globals 
pipeline = None
input_X = None
cosine_sim1 = None
cosine_sim2 = None
cosine_sim3 = None
location_df = None
property_names: List[str] = []
viz_df = None       
db_engine = None    

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, input_X, cosine_sim1, cosine_sim2, cosine_sim3
    global location_df, property_names, viz_df, db_engine

    print(f"Loading models from: {MODELS_DIR}")

    try:
        # ── ML artifacts: download from R2 when configured, else read locally ──
        # All four R2_* vars must be set to enable R2. Missing any one falls back
        # to the local models/ directory, so misconfiguration can't break startup.
        if R2_ENDPOINT_URL and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and R2_BUCKET:
            import boto3
            _s3 = boto3.client(
                "s3",
                endpoint_url=R2_ENDPOINT_URL,
                aws_access_key_id=R2_ACCESS_KEY_ID,
                aws_secret_access_key=R2_SECRET_ACCESS_KEY,
                region_name="auto",
            )
            def _load(key: str):
                obj = _s3.get_object(Bucket=R2_BUCKET, Key=key)
                return pickle.load(io.BytesIO(obj["Body"].read()))
            print(f"Loading artifacts from R2 bucket '{R2_BUCKET}'")
        else:
            def _load(key: str):
                with open(os.path.join(MODELS_DIR, key), "rb") as f:
                    return pickle.load(f)
            print(f"Loading artifacts from local directory: {MODELS_DIR}")

        pipeline    = _load("final_xgb_pipeline.pkl")
        cosine_sim1 = _load("cosine_sim1.pkl")
        cosine_sim2 = _load("cosine_sim2.pkl")
        cosine_sim3 = _load("cosine_sim3.pkl")
        location_df = _load("location_distance.pkl")

        if not (cosine_sim1.shape == cosine_sim2.shape == cosine_sim3.shape ==
                (len(location_df.index), len(location_df.index))):
            raise RuntimeError("Cosine matrices and location_df index size must match.")

        property_names = location_df.index.tolist()
        print(f"Loaded {len(property_names)} properties")

        # Property data: Postgres when DATABASE_URL is set, pkl otherwise 
        if DATABASE_URL:
            from sqlalchemy import create_engine as _create_engine
            # pool_pre_ping keeps connections alive on Neon's serverless backend
            db_engine = _create_engine(
                DATABASE_URL,
                pool_size=5,
                max_overflow=2,
                pool_pre_ping=True,
            )
            input_X = pd.read_sql("SELECT * FROM property_features", db_engine)
            viz_df = pd.read_sql("SELECT * FROM properties_viz", db_engine)
            print(f"Loaded {len(input_X)} feature rows and {len(viz_df)} viz rows from PostgreSQL")
        else:
            with open(os.path.join(MODELS_DIR, "input_data_X.pkl"), "rb") as f:
                input_X = pickle.load(f)
            print("Loaded input features from local pkl (DATABASE_URL not set)")

        print(" Startup complete")
    except Exception as e:
        print(f" Startup failed: {e}")
        raise

    yield

    print("Shutting down...")
    if db_engine:
        db_engine.dispose()

# App 
app = FastAPI(title="Real Estate API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas 
class PriceRequest(BaseModel):
    property_type: str
    sector: str
    bedRoom: float
    bathroom: float
    balcony: str
    agePossession: str
    built_up_area: float = Field(gt=0)
    servant_room: float
    store_room: float
    furnishing_type: str
    luxury_category: str
    floor_category: str

    class Config:
        populate_by_name = True

class PriceResponse(BaseModel):
    price_cr: float

class RecommendRequest(BaseModel):
    property_name: str
    top_n: int = 10
    weight_facilities: float = 30.0
    weight_location: float = 8.0
    weight_price: float = 20.0

class RecommendItem(BaseModel):
    property_name: str
    score: float

class RecommendResponse(BaseModel):
    items: List[RecommendItem]

class LocationSearchRequest(BaseModel):
    location: str
    radius_km: float = Field(gt=0)

class LocationSearchItem(BaseModel):
    property_name: str
    distance_km: float

class LocationSearchResponse(BaseModel):
    items: List[LocationSearchItem]
    count: int

# Open endpoints 
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "message": "Real Estate API",
        "endpoints": {
            "health": "/healthz",
            "predict_price": "/predict-price (POST)",
            "recommend": "/recommend (POST)",
            "properties": "/properties (GET)",
            "sectors": "/sectors (GET)",
            "analytics_data": "/analytics/data (GET)"
        }
    }

# Protected 
@app.get("/properties", dependencies=[Depends(require_api_key)])
def get_properties():
    if property_names:
        return {"properties": property_names, "count": len(property_names)}
    return {"properties": [], "count": 0}

@app.get("/sectors", dependencies=[Depends(require_api_key)])
def get_sectors():
    if input_X is not None:
        sectors = sorted(input_X['sector'].unique().tolist())
        return {"sectors": sectors, "count": len(sectors)}
    return {"sectors": [], "count": 0}

@app.get("/options", dependencies=[Depends(require_api_key)])
def get_options():
    if input_X is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    balconies = input_X['balcony'].unique().tolist()
    balconies_sorted = sorted([b for b in balconies if b != '3+']) + (['3+'] if '3+' in balconies else [])

    return {
        "property_types": sorted(input_X['property_type'].unique().tolist()),
        "sectors": sorted(input_X['sector'].unique().tolist()),
        "bedrooms": sorted(input_X['bedRoom'].unique().tolist()),
        "bathrooms": sorted(input_X['bathroom'].unique().tolist()),
        "balconies": balconies_sorted,
        "age_possession": sorted(input_X['agePossession'].unique().tolist()),
        "furnishing_types": sorted(input_X['furnishing_type'].unique().tolist()),
        "luxury_categories": sorted(input_X['luxury_category'].unique().tolist()),
        "floor_categories": sorted(input_X['floor_category'].unique().tolist())
    }

@app.get("/locations", dependencies=[Depends(require_api_key)])
def get_locations():
    if location_df is not None:
        locations = sorted(location_df.columns.tolist())
        return {"locations": locations, "count": len(locations)}
    return {"locations": [], "count": 0}

# Analytics data 
@app.get("/analytics/data", dependencies=[Depends(require_api_key)])
def get_analytics_data():
    if viz_df is None:
        raise HTTPException(
            status_code=503,
            detail="Analytics data not available — DATABASE_URL is not configured"
        )
    # Use pandas JSON serialiser so NaN becomes null (valid JSON) rather than
    # the bare NaN literal that Python's json module would produce.
    return JSONResponse(content=json.loads(viz_df.to_json(orient="records")))

# Protected – compute endpoints (auth + rate limits) 
@app.post("/predict-price", response_model=PriceResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("10/minute")
def predict_price(request: Request, req: PriceRequest):
    if pipeline is None or input_X is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = [[
        req.property_type,
        req.sector,
        float(req.bedRoom),
        float(req.bathroom),
        req.balcony,
        req.agePossession,
        float(req.built_up_area),
        req.servant_room,
        req.store_room,
        req.furnishing_type,
        req.luxury_category,
        req.floor_category
    ]]

    columns = [
        'property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
        'agePossession', 'built_up_area', 'servant room', 'store room',
        'furnishing_type', 'luxury_category', 'floor_category'
    ]

    X = pd.DataFrame(data, columns=columns)
    y_pred = float(np.expm1(pipeline.predict(X))[0])
    return PriceResponse(price_cr=round(y_pred, 4))

@lru_cache(maxsize=512)
def _cached_recommend(pname: str, top_n: int, w1: float, w2: float, w3: float):
    try:
        idx = property_names.index(pname)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Property '{pname}' not found")

    combo = w1 * cosine_sim1 + w2 * cosine_sim2 + w3 * cosine_sim3
    sims = combo[idx]
    order = np.argsort(-sims)
    top = [i for i in order if i != idx][:top_n]
    return [
        {"property_name": property_names[i], "score": float(sims[i])}
        for i in top
    ]

@app.post("/recommend", response_model=RecommendResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("20/minute")
def recommend(request: Request, req: RecommendRequest):
    if any(v < 0 for v in [req.weight_facilities, req.weight_price, req.weight_location]):
        raise HTTPException(status_code=400, detail="Weights must be non-negative")
    items = _cached_recommend(
        req.property_name,
        req.top_n,
        req.weight_facilities,
        req.weight_price,
        req.weight_location,
    )
    return RecommendResponse(items=items)

@app.post("/search-by-location", response_model=LocationSearchResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
def search_by_location(request: Request, req: LocationSearchRequest):
    if location_df is None:
        raise HTTPException(status_code=503, detail="Location data not loaded")

    if req.location not in location_df.columns:
        raise HTTPException(status_code=404, detail=f"Location '{req.location}' not found")

    radius_m = req.radius_km * 1000
    result_ser = location_df[location_df[req.location] < radius_m][req.location].sort_values()

    items = [
        LocationSearchItem(
            property_name=str(key),
            distance_km=round(value / 1000, 2)
        )
        for key, value in result_ser.items()
    ]

    return LocationSearchResponse(items=items, count=len(items))
