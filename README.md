# Boston Smart Accident Risk and Rerouting System

**Predicting accident risk and recommending safer routes using ML, historical crash data, and live route and weather data integration.**

## Overview

This project combines machine learning, real-time data integration, and route optimization to help drivers avoid high-risk areas and find safer alternative routes. By analyzing 47,000+ Boston crash records (2015–2024) with spatial features, weather patterns, and traffic conditions, the system estimates accident/disruption risk for any route and recommends low-risk alternatives.

The system is deployed as a live REST API with a web interface, making real-time risk prediction accessible to end users and data scientists alike.

## Problem Solved

**Challenge:** Drivers in Boston have no easy way to understand accident risk on their planned routes or discover safer alternatives before leaving.

**Solution:** 
- **Risk Scoring:** Machine learning models estimate accident probability for routes based on location, time, weather, and historical patterns
- - **Safer Alternatives:** The API suggests lower-risk alternate routes using real-time routing data
  - - **Data-Driven Insights:** Historical analysis reveals accident hotspots, seasonal trends, and weather-related risk patterns
    - - **Accessible API:** No CSV downloads required—query the database directly via REST endpoints
     
      - ## System Architecture
     
      - ```
        ┌─────────────────────────────────────────────────────────────────┐
        │                     FRONTEND (Web UI)                            │
        │                    static/index.html                            │
        └────────────────────────────┬────────────────────────────────────┘
                                     │
        ┌────────────────────────────▼────────────────────────────────────┐
        │                    FASTAPI BACKEND                              │
        │                      (api.py)                                   │
        │  ┌──────────────────────────────────────────────────────────┐  │
        │  │ Routes:                                                  │  │
        │  │ - /crashes (list, filter, aggregate)                    │  │
        │  │ - /stats (yearly crash trends)                          │  │
        │  │ - /predict (route risk prediction)                      │  │
        │  │ - /hotspots (accident hotspot locations)                │  │
        │  └──────────────────────────────────────────────────────────┘  │
        └───┬──────────┬──────────────┬──────────────┬──────────────┬────┘
            │          │              │              │              │
            ▼          ▼              ▼              ▼              ▼
        ┌────────┐ ┌──────────┐ ┌────────────┐ ┌─────────┐ ┌──────────┐
        │Database│ │  Models  │ │  Features  │ │ Weather │ │  Routes  │
        │Supabase│ │models/   │ │ Engineered │ │ (live)  │ │ (live)   │
        │        │ │ .pkl     │ │ via src/   │ │ APIs    │ │ APIs     │
        └────────┘ └──────────┘ └────────────┘ └─────────┘ └──────────┘
        ```

        **Key Components:**

        1. **API Backend (FastAPI)** — RESTful endpoints for crash data, predictions, and hotspot analysis
        2. 2. **ML Models** — Versioned serialized models (v1–v4) predicting accident risk from features
           3. 3. **Feature Engineering** — Spatial, temporal, and weather-based features extracted from historical data
              4. 4. **Live Integrations** — Real-time weather and routing API calls to enrich predictions
                 5. 5. **Database (Supabase)** — PostgreSQL backend storing 47,000+ crash records with spatial indexing
                    6. 6. **Web UI** — Single-page app for interactive route planning and risk visualization
                      
                       7. ## Tech Stack
                      
                       8. | Component | Technology | Purpose |
                       9. |-----------|-----------|---------|
                       10. | **Backend Framework** | FastAPI + Uvicorn | REST API server, async request handling |
                       11. | **Database** | Supabase (PostgreSQL) | 47,000+ crash records, queries via REST |
                       12. | **ML Models** | scikit-learn, LightGBM, imbalanced-learn | Accident risk classification, SMOTE resampling |
                       13. | **Feature Engineering** | Pandas, NumPy, Polyline | Data preprocessing, geospatial encoding |
                       14. | **Live Data** | OpenWeather API, OpenRouteService | Real-time weather/traffic for predictions |
                       15. | **Geocoding** | Geopy / OpenCage | Address → coordinates conversion |
                       16. | **Deployment** | Render (free tier) | Live API hosting |
                       17. | **Testing** | pytest (unit tests) | Model, geocoding, routes, weather modules |
                       18. | **Frontend** | HTML/CSS/JS (static/) | Web UI for route prediction |
                      
                       19. ## Key Features
                      
                       20. ✅ **Multi-Version Model Selection** — Models v1–v4 with increasing sophistication (spatial features, SMOTE balancing)
                       21. ✅ **Route Segmentation Risk** — Break routes into segments, predict risk for each, aggregate scores
                       22. ✅ **Crash Hotspot Detection** — Identify EMS-flagged high-risk intersection clusters
                       23. ✅ **Historical Filtering** — Query crashes by year, city, severity, weather conditions (2015–2024)
                       24. ✅ **Accident Aggregation** — Yearly trends, injury/fatality counts, time-series patterns
                       25. ✅ **Live Weather Integration** — Adjust risk predictions based on current weather
                       26. ✅ **Alternate Route Ranking** — Compare multiple routes, recommend safest option
                       27. ✅ **Unit Test Coverage** — Automated tests for geocoding, prediction, routes, weather modules
                      
                       28. ## How to Run Locally
                      
                       29. ### Prerequisites
                      
                       30. ```bash
                           # Python 3.9+
                           python --version

                           # Clone repository
                           git clone https://github.com/machackgo/boston-smart-accident-risk-rerouting.git
                           cd boston-smart-accident-risk-rerouting
                           ```

                           ### Setup

                           1. **Install dependencies:**
                           2. ```bash
                              pip install -r requirements.txt
                              ```

                              2. **Environment variables** — Copy `.env.example` to `.env` and fill in:
                              3. ```bash
                                 cp .env.example .env
                                 # Edit .env with your API keys:
                                 # - SUPABASE_URL
                                 # - SUPABASE_KEY
                                 # - OPENWEATHER_API_KEY
                                 # - OPENROUTESERVICE_API_KEY (optional, for live routing)
                                 ```

                                 3. **Run API server:**
                                 4. ```bash
                                    uvicorn api:app --reload --host 0.0.0.0 --port 8000
                                    ```

                                    4. **Access:**
                                    5. - **Swagger docs:** http://localhost:8000/docs
                                       - - **Web UI:** http://localhost:8000 (if serving static/)
                                        
                                         - 5. **Run tests:**
                                           6. ```bash
                                              pytest tests/
                                              ```

                                              ## Skills Demonstrated

                                              | Skill | Evidence |
                                              |-------|----------|
                                              | **Machine Learning Engineering** | Multiple model versions with hyperparameter tuning, class imbalance handling (SMOTE), feature selection |
                                              | **Data Engineering** | 47K-row crash dataset ingestion, geospatial feature extraction, caching strategies (Parquet) |
                                              | **Backend API Design** | FastAPI async routing, Pydantic request validation, error handling, CORS middleware |
                                              | **Database Design** | Schema modeling in PostgreSQL, querying via Supabase REST API, efficient indexing |
                                              | **Feature Engineering** | Temporal features (hour, day, month, seasonality), spatial features (lat/lon encoding), weather integration |
                                              | **Real-Time Integration** | Live weather API calls, dynamic routing with OpenRouteService, async request handling |
                                              | **Testing & QA** | Unit tests for geocoding, prediction, routes, and weather modules |
                                              | **Deployment** | Render free-tier hosting, environment configuration, cold start optimization |

                                              ## VeriBridge Proof Evidence

                                              | Skill | Evidence File | Location | Details |
                                              |-------|---------------|----------|---------|
                                              | **ML Model Serving** | `api.py` | Lines 150–200 (predict endpoints) | Real-time risk prediction via `/predict` endpoints, model versioning with fallback logic |
                                              | **Feature Engineering** | `src/predict/feature_builder.py` | Full module | Spatial, temporal, and weather feature extraction from raw crash data |
                                              | **Accident Risk Classification** | `models/best_model_v4.pkl` | Serialized model | LightGBM classifier trained on 47K crash records with SMOTE balancing |
                                              | **Route Risk Aggregation** | `api.py` | Lines 175–190 | `predict_route_risk_segmented()` breaks routes into segments and aggregates risk scores |
                                              | **Live Weather Integration** | `src/live/weather.py` | Full module | OpenWeather API calls to enrich route predictions with real-time conditions |
                                              | **Database Integration** | `api.py` | Lines 30–50 | Supabase REST client querying 47K crashes, filtering by year/city/severity/weather |
                                              | **Alternative Route Finding** | `src/live/routes.py` | Full module | Queries OpenRouteService for alternate routes, supports route segmentation |
                                              | **Hotspot Detection** | `api.py` | GET `/crashes/hotspots` | EMS-flagged crash locations, clustering analysis |
                                              | **Test Coverage** | `tests/` | 4 modules | Unit tests for geocoding, prediction, routes, weather integrations |
                                              | **REST API Design** | `api.py` | Full module | Async FastAPI endpoints, Pydantic validation, error handling, CORS |

                                              ## Recruiter Value

                                              **Why This Project Matters:**

                                              1. **Real-World Problem** — Addresses actual traffic safety in Boston using public data
                                              2. 2. **Full ML Pipeline** — Data ingestion → feature engineering → model training → inference → deployment
                                                 3. 3. **Scalable Architecture** — FastAPI + Supabase can handle thousands of concurrent users
                                                    4. 4. **Production Deployment** — Live API accessible at https://boston-smart-accident-risk-rerouting.onrender.com
                                                       5. 5. **Multi-Model Strategy** — Demonstrates iterative improvement (v1–v4) with A/B testing considerations
                                                          6. 6. **Testing Discipline** — Unit tests for all critical modules ensure code quality
                                                             7. 7. **API-First Design** — REST endpoints make ML accessible to non-technical end users
                                                                8. 8. **Real-Time Features** — Integrates live weather and routing, not just static historical data
                                                                  
                                                                   9. ## Live Deployment
                                                                  
                                                                   10. **API Status:** ✅ Live at https://boston-smart-accident-risk-rerouting.onrender.com
                                                                  
                                                                   11. **Quick Test:**
                                                                   12. ```bash
                                                                       curl "https://boston-smart-accident-risk-rerouting.onrender.com/crashes/fatal?limit=5"
                                                                       ```

                                                                       ⚠️ **Note:** First request may take 30–60 seconds due to Render free tier cold start. Subsequent requests complete in <500ms.

                                                                       **Swagger Documentation:** https://boston-smart-accident-risk-rerouting.onrender.com/docs

                                                                       ## Future Improvements

                                                                       - [ ] **Docker containerization** — Simplify deployment and local development
                                                                       - [ ] - [ ] **CI/CD pipeline** — Automate testing and deployment on GitHub Actions
                                                                       - [ ] - [ ] **Model retraining pipeline** — Automated weekly/monthly retraining with new crash data
                                                                       - [ ] - [ ] **User authentication** — API keys for rate-limited access
                                                                       - [ ] - [ ] **Advanced visualization** — Interactive heat maps of risk zones
                                                                       - [ ] - [ ] **Mobile app** — React Native app for iOS/Android with native routing integration
                                                                       - [ ] - [ ] **Reinforcement learning** — Optimize route recommendations based on actual outcomes
                                                                       - [ ] - [ ] **Price/time tradeoffs** — Allow users to balance risk vs. travel time
                                                                       - [ ] - [ ] **Traffic incident correlation** — Integrate live traffic incident feeds for dynamic risk updates
                                                                      
                                                                       - [ ] ## References
                                                                      
                                                                       - [ ] - **Dataset:** MassDOT Boston Crash Database (2015–2024), 47,000+ records
                                                                       - [ ] - **Models:** LightGBM, scikit-learn (SMOTE for class imbalance)
                                                                       - [ ] - **APIs:** OpenWeather, OpenRouteService, Geopy
                                                                       - [ ] - **Deployment:** Render (free tier)
                                                                      
                                                                       - [ ] ---
                                                                      
                                                                       - [ ] **Team:** Mohammed Mubashir Uddin Faraz, Sandhia Maheshwari, Himabindu Tummala, Kamal Dalal
                                                                       - [ ] **Project Type:** Intro to Data Science (Worcester Polytechnic Institute)
                                                                       - [ ] **Repository:** https://github.com/machackgo/boston-smart-accident-risk-rerouting
                                                                       - [ ] **Live API:** https://boston-smart-accident-risk-rerouting.onrender.com
