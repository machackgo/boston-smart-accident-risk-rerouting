# Boston Smart Accident Risk and Rerouting System

Team project for Intro to Data Science.

## Team Members
- Mohammed Mubashir Uddin Faraz
- Sandhia Maheshwari
- Himabindu Tummala
- Kamal Dalal

## Project Idea
This project uses historical Boston accident data, live weather, live traffic conditions, and route information to estimate accident/disruption risk and recommend alternate routes.

## Dataset API

We uploaded the full Boston crash dataset (47,000+ rows, 2015–2024) from MassDOT to a live REST API — no CSV download needed. The API is built with **FastAPI**, hosted on **Render**, and backed by **Supabase** as the database.

**Live API docs:** https://boston-smart-accident-risk-rerouting.onrender.com/docs

### Available Endpoints

| Endpoint | Description |
|---|---|
| `GET /crashes` | All crashes (paginated) |
| `GET /crashes/year/{year}` | Filter by year (2015–2024) |
| `GET /crashes/city/{city}` | Filter by city/town name |
| `GET /crashes/severity/{severity}` | Filter by severity class |
| `GET /crashes/fatal` | Crashes with at least 1 fatality |
| `GET /crashes/hotspots` | EMS-flagged crash hotspot locations |
| `GET /crashes/filter` | Multi-field filter (year, city, severity, weather) |
| `GET /stats/by-year` | Aggregated crash counts and injuries per year |

### Example: Use the API in Jupyter

```python
import requests

BASE = "https://boston-smart-accident-risk-rerouting.onrender.com"

# Get all fatal crashes
response = requests.get(f"{BASE}/crashes/fatal", params={"limit": 50})
data = response.json()["data"]

# Get crashes in 2020 with rain
response = requests.get(f"{BASE}/crashes/filter", params={
    "year": 2020,
    "weather": "Rain",
    "limit": 100
})
df = pd.DataFrame(response.json()["data"])
df.head()
```

No CSV download required — query only what you need, directly into a DataFrame.
