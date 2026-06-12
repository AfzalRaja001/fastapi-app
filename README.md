# Real Estate Analytics API

A high-performance FastAPI backend for property price prediction and recommendation.

**Live API Root:** [https://real-estate-fastapi-latest.onrender.com/](https://real-estate-fastapi-latest.onrender.com/)  
**Interactive Docs (Swagger UI):** [https://real-estate-fastapi-latest.onrender.com/docs](https://real-estate-fastapi-latest.onrender.com/docs)  

**Docker Image:** [docker.io/afzal023/real-estate-fastapi:latest](https://hub.docker.com/r/afzal023/real-estate-fastapi)

## Features

- 🚀 **FastAPI** based high-performance server
- 🤖 **XGBoost** integration for price prediction
- 🔍 **Cosine Similarity** based recommendation engine
- 🗺️ **Geospatial Search** for location-based queries
- 🐳 **Dockerized** for easy deployment

## Deployment

### Docker Deployment

**Pull the image:**
```bash
docker pull afzal023/real-estate-fastapi:latest
```

**Run the container:**
```bash
docker run -p 8000:8000 afzal023/real-estate-fastapi:latest
```

**Deploying on Render:**
1. Create a new **Web Service**.
2. Select **"Deploy an existing image from a registry"**.
3. Image URL: `afzal023/real-estate-fastapi:latest`
4. Port: `8000` (Render detects this, or you can set `PORT` env var).

## API Endpoints

### General
- `GET /` - API Info
- `GET /healthz` - Health Check

### Property Data
- `GET /properties` - List all properties
- `GET /sectors` - List available sectors
- `GET /locations` - List searchable locations
- `GET /options` - Get form dropdown options

### Predictions & Recommendations
- `POST /predict-price` - Predict property price based on features.
- `POST /recommend` - Get similar property recommendations.
- `POST /search-by-location` - Find properties within a radius.

## Local Development

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

**Run Server:**
```bash
uvicorn app:app --reload
```
Server runs at `http://localhost:8000`
