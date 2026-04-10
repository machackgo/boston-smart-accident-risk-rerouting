"""
Step 2: FastAPI for Boston Crash Data
Run: pip install fastapi uvicorn supabase
Then: uvicorn api:app --reload
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, Query
from supabase import create_client
from typing import Optional

# ── Credentials ──────────────────────────────────────────────
SUPABASE_URL = "https://iizfaawqzzrnhfaihimp.supabase.co"
SUPABASE_KEY = "sb_publishable_EnEsuHwXoNl4bQLeM2221A_LBW4nnNO"  # publishable key for reading

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI(title="Boston Crash Data API", version="1.0")

TABLE = "boston_crashes"

# ── 1. Get all crashes (paginated) ───────────────────────────
@app.get("/crashes")
def get_crashes(limit: int = 100, offset: int = 0):
    """Get crashes with pagination. Default 100 per page."""
    res = supabase.table(TABLE).select("*").range(offset, offset + limit - 1).execute()
    return {"total_returned": len(res.data), "data": res.data}

# ── 2. Filter by year ─────────────────────────────────────────
@app.get("/crashes/year/{year}")
def get_by_year(year: int, limit: int = 100):
    """Get crashes for a specific year (2015–2024)."""
    res = supabase.table(TABLE).select("*").eq("year", year).limit(limit).execute()
    return {"year": year, "total_returned": len(res.data), "data": res.data}

# ── 3. Filter by severity ─────────────────────────────────────
@app.get("/crashes/severity/{severity}")
def get_by_severity(severity: str, limit: int = 100):
    """
    Severity options: Fatal, Non-fatal injury, Property damage only
    Use SEVERITY_3CLASS values.
    """
    res = supabase.table(TABLE).select("*").ilike("severity_3class", f"%{severity}%").limit(limit).execute()
    return {"severity": severity, "total_returned": len(res.data), "data": res.data}

# ── 4. Filter by city ─────────────────────────────────────────
@app.get("/crashes/city/{city}")
def get_by_city(city: str, limit: int = 100):
    """Get crashes in a specific city/town."""
    res = supabase.table(TABLE).select("*").ilike("city_town_name", f"%{city}%").limit(limit).execute()
    return {"city": city, "total_returned": len(res.data), "data": res.data}

# ── 5. Get hotspots ───────────────────────────────────────────
@app.get("/crashes/hotspots")
def get_hotspots(limit: int = 100):
    """Get crash hotspot locations (ems_hotspot_flag = 1)."""
    res = supabase.table(TABLE).select("*").eq("ems_hotspot_flag", 1).limit(limit).execute()
    return {"total_returned": len(res.data), "data": res.data}

# ── 6. Fatal crashes only ─────────────────────────────────────
@app.get("/crashes/fatal")
def get_fatal(limit: int = 100):
    """Get crashes with at least 1 fatality."""
    res = supabase.table(TABLE).select("*").gt("numb_fatal_injr", 0).limit(limit).execute()
    return {"total_returned": len(res.data), "data": res.data}

# ── 7. Summary stats by year ──────────────────────────────────
@app.get("/stats/by-year")
def stats_by_year():
    """Get crash counts grouped by year."""
    res = supabase.table(TABLE).select("year, numb_fatal_injr, numb_nonfatal_injr").execute()
    from collections import defaultdict
    stats = defaultdict(lambda: {"crashes": 0, "fatalities": 0, "injuries": 0})
    for row in res.data:
        y = row["year"]
        stats[y]["crashes"]    += 1
        stats[y]["fatalities"] += (row["numb_fatal_injr"] or 0)
        stats[y]["injuries"]   += (row["numb_nonfatal_injr"] or 0)
    return {"data": dict(sorted(stats.items()))}

# ── 8. Advanced filter ────────────────────────────────────────
@app.get("/crashes/filter")
def filter_crashes(
    year:     Optional[int] = Query(None, description="e.g. 2020"),
    city:     Optional[str] = Query(None, description="e.g. Boston"),
    severity: Optional[str] = Query(None, description="e.g. Fatal"),
    weather:  Optional[str] = Query(None, description="e.g. Rain"),
    limit:    int = 100
):
    """Filter by multiple fields at once."""
    q = supabase.table(TABLE).select("*")
    if year:     q = q.eq("year", year)
    if city:     q = q.ilike("city_town_name", f"%{city}%")
    if severity: q = q.ilike("severity_3class", f"%{severity}%")
    if weather:  q = q.ilike("weath_cond_descr", f"%{weather}%")
    res = q.limit(limit).execute()
    return {"filters_applied": {"year": year, "city": city, "severity": severity, "weather": weather},
            "total_returned": len(res.data), "data": res.data}
