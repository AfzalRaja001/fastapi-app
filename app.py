import os
import pickle
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Config 
base_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get(
    "MODELS_DIR",
    os.path.join(base_dir, "models")
)
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")

# Artifacts Loading - Global variables
pipeline = None
input_X = None
cosine_sim1 = None
cosine_sim2 = None
cosine_sim3 = None
location_df = None
property_names: List[str] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup: Load artifacts
    global pipeline, input_X, cosine_sim1, cosine_sim2, cosine_sim3, location_df, property_names
    
    print(f"Loading models from: {MODELS_DIR}")
    
    try:
        # Price model
        with open(os.path.join(MODELS_DIR, "final_xgb_pipeline.pkl"), "rb") as f:
            pipeline = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "input_data_X.pkl"), "rb") as f:
            input_X = pickle.load(f)

        # Recommender artifacts
        with open(os.path.join(MODELS_DIR, "cosine_sim1.pkl"), "rb") as f:
            cosine_sim1 = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "cosine_sim2.pkl"), "rb") as f:
            cosine_sim2 = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "cosine_sim3.pkl"), "rb") as f:
            cosine_sim3 = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "location_distance.pkl"), "rb") as f:
            location_df = pickle.load(f)

        if not (cosine_sim1.shape == cosine_sim2.shape == cosine_sim3.shape == (len(location_df.index), len(location_df.index))):
            raise RuntimeError("Cosine matrices and location_df index size must match.")

        property_names = location_df.index.tolist()
        
        print(f"✅ Loaded {len(property_names)} properties")
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise
    
    yield
    
    # Shutdown: Cleanup if needed
    print("Shutting down...")

# App with lifespan
app = FastAPI(title="Real Estate API", lifespan=lifespan)
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
            "sectors": "/sectors (GET)"
        }
    }

@app.get("/properties")
def get_properties():
    """Get list of all available properties"""
    if property_names:
        return {"properties": property_names, "count": len(property_names)}
    return {"properties": [], "count": 0}

@app.get("/sectors")
def get_sectors():
    """Get list of all available sectors"""
    if input_X is not None:
        sectors = sorted(input_X['sector'].unique().tolist())
        return {"sectors": sectors, "count": len(sectors)}
    return {"sectors": [], "count": 0}

@app.get("/options")
def get_options():
    """Get all available options for dropdowns"""
    if input_X is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Custom sorting for balconies to handle '3+' properly
    balconies = input_X['balcony'].unique().tolist()
    balconies_sorted = sorted([b for b in balconies if b != '3+']) + (['3+'] if '3+' in balconies else [])
    
    return {
        "property_types": sorted(input_X['property_type'].unique().tolist()),
        "sectors": sorted(input_X['sector'].unique().tolist()),
        "bedrooms": sorted(input_X['bedRoom'].unique().tolist()),
        "bathrooms": sorted(input_X['bathroom'].unique().tolist()),
        "balconies": balconies_sorted,  # ['0', '1', '2', '3', '3+']
        "age_possession": sorted(input_X['agePossession'].unique().tolist()),
        "furnishing_types": sorted(input_X['furnishing_type'].unique().tolist()),
        "luxury_categories": sorted(input_X['luxury_category'].unique().tolist()),
        "floor_categories": sorted(input_X['floor_category'].unique().tolist())
    }

# Price endpoints
@app.post("/predict-price", response_model=PriceResponse)
def predict_price(req: PriceRequest):
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

# Recommendation endpoints
@lru_cache(maxsize=512)
def _cached_recommend(pname: str, top_n: int, w1: float, w2: float, w3: float):
    try:
        idx = property_names.index(pname)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Property '{pname}' not found")

    combo = w1 * cosine_sim1 + w2 * cosine_sim2 + w3 * cosine_sim3
    sims = combo[idx]

    # Top-N excluding self
    order = np.argsort(-sims)
    top = [i for i in order if i != idx][:top_n]
    items = [
        {"property_name": property_names[i], "score": float(sims[i])}
        for i in top
    ]
    return items

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
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

# Location search
class LocationSearchRequest(BaseModel):
    location: str
    radius_km: float = Field(gt=0)

class LocationSearchItem(BaseModel):
    property_name: str
    distance_km: float

class LocationSearchResponse(BaseModel):
    items: List[LocationSearchItem]
    count: int

@app.post("/search-by-location", response_model=LocationSearchResponse)
def search_by_location(req: LocationSearchRequest):
    """Search properties within a radius from a location"""
    if location_df is None:
        raise HTTPException(status_code=503, detail="Location data not loaded")
    
    if req.location not in location_df.columns:
        raise HTTPException(status_code=404, detail=f"Location '{req.location}' not found")
    
    # Filter properties within radius (distance in meters, convert to km)
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

@app.get("/locations")
def get_locations():
    """Get list of all available locations for search"""
    if location_df is not None:
        locations = sorted(location_df.columns.tolist())
        return {"locations": locations, "count": len(locations)}
    return {"locations": [], "count": 0}