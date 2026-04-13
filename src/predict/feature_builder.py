"""
feature_builder.py — Assembles the 80-feature row required by best_model_v2.pkl.

Orchestrates three live API calls:
    1. get_route()   — route geometry + midpoint coordinates + travel time
    2. get_weather() — current weather at the route midpoint
  (geocoding is handled implicitly inside get_route for address inputs)

Then maps live API responses onto the exact v2 feature schema, in the exact
column order read from models/feature_list_v2.txt.

OpenWeather condition → v2 weath_cond_descr_* mapping:
    "Clear"                         → weath_cond_descr_Clear
    "Clouds"                        → weath_cond_descr_Cloudy
    "Rain" / "Drizzle"              → weath_cond_descr_Rain
    "Snow"                          → weath_cond_descr_Snow
    "Fog" / "Mist" / "Haze" /
        "Smoke" / "Dust" / "Sand"   → weath_cond_descr_Fog__smog__smoke
    "Thunderstorm" / "Squall" /
        "Tornado"                   → weath_cond_descr_Severe_crosswinds
    anything else                   → weath_cond_descr_Not_Reported
"""

import sys
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd

BOSTON_TZ = ZoneInfo("America/New_York")

# Allow imports from src/ regardless of working directory
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from live.routes import get_route
from live.weather import get_weather

# Path to the v2 feature list — read once at import time
_FEATURE_LIST_PATH = _REPO_ROOT / "models" / "feature_list_v2.txt"

with open(_FEATURE_LIST_PATH) as _f:
    V2_FEATURES = [line.strip() for line in _f if line.strip()]

def _smarter_speed_default(distance_miles: float) -> float:
    """Distance-based speed limit heuristic when live data is unavailable."""
    if distance_miles < 2.0:
        return 25.0
    if distance_miles < 10.0:
        return 35.0
    return 55.0

# OpenWeather main condition → v2 feature column
_WEATHER_MAP = {
    "Clear":        "weath_cond_descr_Clear",
    "Clouds":       "weath_cond_descr_Cloudy",
    "Rain":         "weath_cond_descr_Rain",
    "Drizzle":      "weath_cond_descr_Rain",
    "Snow":         "weath_cond_descr_Snow",
    "Fog":          "weath_cond_descr_Fog__smog__smoke",
    "Mist":         "weath_cond_descr_Fog__smog__smoke",
    "Haze":         "weath_cond_descr_Fog__smog__smoke",
    "Smoke":        "weath_cond_descr_Fog__smog__smoke",
    "Dust":         "weath_cond_descr_Fog__smog__smoke",
    "Sand":         "weath_cond_descr_Fog__smog__smoke",
    "Ash":          "weath_cond_descr_Fog__smog__smoke",
    "Thunderstorm": "weath_cond_descr_Severe_crosswinds",
    "Squall":       "weath_cond_descr_Severe_crosswinds",
    "Tornado":      "weath_cond_descr_Severe_crosswinds",
}
_WEATHER_FALLBACK = "weath_cond_descr_Not_Reported"


def _resolve_time(departure_time):
    """
    Return a datetime localised to America/New_York (handles EDT/EST automatically).

    Rules:
      - None              → current wall-clock time in Boston
      - aware datetime    → convert to Boston tz
      - naive datetime    → treat as already Boston local time
      - aware ISO string  → parse then convert to Boston tz
      - naive ISO string  → treat as Boston local time
    """
    if departure_time is None:
        return datetime.now(BOSTON_TZ)
    if isinstance(departure_time, datetime):
        if departure_time.tzinfo:
            return departure_time.astimezone(BOSTON_TZ)
        return departure_time.replace(tzinfo=BOSTON_TZ)
    # ISO string
    dt = datetime.fromisoformat(departure_time)
    if dt.tzinfo:
        return dt.astimezone(BOSTON_TZ)
    return dt.replace(tzinfo=BOSTON_TZ)


def _time_features(dt: datetime) -> dict:
    """Compute all time-derived features from a datetime."""
    hour        = dt.hour
    dow         = dt.weekday()   # 0=Monday
    is_weekend  = int(dow >= 5)
    is_rush     = int((hour in range(7, 10) or hour in range(16, 20)) and not is_weekend)

    return {
        "hour_of_day": hour,
        "day_of_week": dow,
        "month":       dt.month,
        "is_weekend":  is_weekend,
        "is_rush_hour": is_rush,
    }


def _light_phase_features(hour: int) -> dict:
    """Engineer light-phase flags from hour of day (NOT from ambnt_light_descr)."""
    if 7 <= hour <= 18:
        return {"light_phase_Daylight": 1, "light_phase_Dawn_Dusk": 0, "light_phase_Dark": 0}
    if hour in (5, 6, 19, 20):
        return {"light_phase_Daylight": 0, "light_phase_Dawn_Dusk": 1, "light_phase_Dark": 0}
    return {"light_phase_Daylight": 0, "light_phase_Dawn_Dusk": 0, "light_phase_Dark": 1}


def _map_weather(ow_condition: str) -> tuple[str, str]:
    """
    Map an OpenWeather condition string to a v2 feature column name.

    Returns:
        (feature_column_name, human_readable_mapping_description)
    """
    col = _WEATHER_MAP.get(ow_condition, _WEATHER_FALLBACK)
    desc = f"'{ow_condition}' → {col}"
    return col, desc


def build_segment_features(
    route: dict,
    sample_points: list,
    departure_time=None,
    speed_limits_per_point: list | None = None,
) -> tuple:
    """
    Build an N-row feature DataFrame for a list of sample points along a route.

    Uses ONE weather call for the entire route (weather doesn't change every 500 m)
    and applies shared time features to all rows.  Only lat/lng (and speed_limit)
    differ per row.

    Args:
        route (dict): Return value of get_route() — must contain default_route.
        sample_points (list): List of (lat, lng) tuples, one per segment sample.
        departure_time: ISO 8601 string, datetime object, or None (= now).
        speed_limits_per_point (list | None): Per-point speed limit estimates
            (same length as sample_points).  If None, falls back to the
            distance-based heuristic.

    Returns:
        features_df (pd.DataFrame): N-row DataFrame in exact V2_FEATURES column order.
        context (dict): Transparency dict — weather used, time features, num points.
    """
    dt = _resolve_time(departure_time)
    print(f"[feature_builder] segment mode | Boston time: {dt.strftime('%Y-%m-%d %H:%M %Z')}, "
          f"points={len(sample_points)}")

    # ── Weather: one call at route midpoint ───────────────────────────────────
    default_route = route["default_route"]
    midpoint      = default_route["midpoint_coords"]
    if midpoint is None:
        raise ValueError("Route has no decoded polyline — cannot fetch weather.")
    mid_lat = midpoint["lat"]
    mid_lng = midpoint["lng"]

    print(f"[feature_builder] Fetching weather at midpoint ({mid_lat}, {mid_lng}) ...")
    weather       = get_weather(lat=mid_lat, lng=mid_lng)
    ow_condition  = weather["condition"]
    weather_col, weather_mapping_desc = _map_weather(ow_condition)
    print(f"[feature_builder] Weather mapping: {weather_mapping_desc}")

    # ── Time features: shared across all points ───────────────────────────────
    time_feats  = _time_features(dt)
    light_feats = _light_phase_features(time_feats["hour_of_day"])

    # Fallback speed (distance-based) used when per-point data is absent
    route_dist = default_route.get("distance_miles", 5.0)
    fallback_speed = _smarter_speed_default(route_dist)

    # ── Build one row per sample point ────────────────────────────────────────
    rows = []
    for i, (lat, lng) in enumerate(sample_points):
        row = {feat: 0 for feat in V2_FEATURES}
        row["lat"]         = lat
        row["lon"]         = lng

        # Per-point speed limit (from Google traffic intervals or distance fallback)
        if speed_limits_per_point and i < len(speed_limits_per_point) and speed_limits_per_point[i] is not None:
            row["speed_limit"] = speed_limits_per_point[i]
        else:
            row["speed_limit"] = fallback_speed

        row.update(time_feats)
        row.update(light_feats)
        if weather_col in row:
            row[weather_col] = 1
        else:
            row[_WEATHER_FALLBACK] = 1
        rows.append(row)

    features_df = pd.DataFrame(rows)[V2_FEATURES]

    context = {
        "weather_raw":          weather,
        "weather_mapping_used": weather_mapping_desc,
        "weather_col":          weather_col,
        "departure_time":       dt.isoformat(),
        "time_features":        time_feats,
        "local_hour":           time_feats["hour_of_day"],
        "num_points":           len(sample_points),
        "speed_source":         "per_segment" if speed_limits_per_point else "distance_heuristic",
        "fallback_speed_mph":   fallback_speed,
    }

    return features_df, context


def build_features(origin: str, destination: str, departure_time=None) -> tuple:
    """
    Orchestrate live API calls and assemble the 80-feature DataFrame row
    required by best_model_v2.pkl.

    Args:
        origin (str): Address or place name for the start of the route.
        destination (str): Address or place name for the end of the route.
        departure_time: ISO 8601 string, datetime object, or None (= now).

    Returns:
        features_df (pd.DataFrame): Single-row DataFrame with all 80 v2 features
                                    in the exact order from feature_list_v2.txt.
        context (dict): Transparency dict with route info, weather info, and
                        mapping decisions used to build the feature row.

    Raises:
        EnvironmentError: If a required API key is missing.
        requests.ConnectionError: On network failures.
        requests.HTTPError: On non-200 API responses.
        ValueError: If the route API returns no results.
    """
    dt = _resolve_time(departure_time)
    print(f"[feature_builder] Local Boston time: {dt.strftime('%Y-%m-%d %H:%M %Z')}, hour={dt.hour}")

    # ── Step 1: Route (also implicitly geocodes the addresses) ────────────────
    print(f"[feature_builder] Fetching route: '{origin}' → '{destination}' ...")
    route = get_route(origin, destination)

    default_route = route["default_route"]
    midpoint      = default_route["midpoint_coords"]   # {"lat": ..., "lng": ...}

    if midpoint is None:
        raise ValueError(
            "Route API returned no decoded polyline — cannot determine midpoint. "
            "Check that the origin/destination are valid driving locations."
        )

    mid_lat = midpoint["lat"]
    mid_lng = midpoint["lng"]
    print(f"[feature_builder] Route midpoint: lat={mid_lat}, lng={mid_lng}")

    # ── Step 2: Weather at midpoint ───────────────────────────────────────────
    print(f"[feature_builder] Fetching weather at midpoint ...")
    weather = get_weather(lat=mid_lat, lng=mid_lng)
    ow_condition = weather["condition"]   # e.g. "Clear", "Rain", "Snow"

    # ── Step 3: Map weather to v2 feature ─────────────────────────────────────
    weather_col, weather_mapping_desc = _map_weather(ow_condition)
    print(f"[feature_builder] Weather mapping: {weather_mapping_desc}")

    # ── Step 4: Time features ─────────────────────────────────────────────────
    time_feats  = _time_features(dt)
    light_feats = _light_phase_features(time_feats["hour_of_day"])

    # ── Step 5: Assemble feature dict — start with all zeros ─────────────────
    row = {feat: 0 for feat in V2_FEATURES}

    # Location
    row["lat"] = mid_lat
    row["lon"] = mid_lng

    # Road property — use distance-based heuristic (no per-segment data in non-segmented path)
    row["speed_limit"] = _smarter_speed_default(default_route["distance_miles"])

    # Time
    row.update(time_feats)

    # Light phase
    row.update(light_feats)

    # Weather (only the mapped column gets a 1; all others stay 0)
    if weather_col in row:
        row[weather_col] = 1
    else:
        # Fallback if sanitized name doesn't match exactly
        row[_WEATHER_FALLBACK] = 1
        weather_mapping_desc += f" (fallback to {_WEATHER_FALLBACK}: column not in schema)"

    # ── Step 6: Build DataFrame in exact V2_FEATURES order ───────────────────
    features_df = pd.DataFrame([row])[V2_FEATURES]

    # ── Step 7: Build context dict ────────────────────────────────────────────
    context = {
        "origin":           origin,
        "destination":      destination,
        "departure_time":   dt.isoformat(),
        "midpoint_lat":     mid_lat,
        "midpoint_lng":     mid_lng,
        "hour_of_day":      time_feats["hour_of_day"],
        "is_rush_hour":     time_feats["is_rush_hour"],
        "is_weekend":       time_feats["is_weekend"],
        "weather_condition_raw":  ow_condition,
        "weather_mapping_used":   weather_mapping_desc,
        "weather_feature_set":    weather_col,
        "light_phase":      next(k for k, v in light_feats.items() if v == 1),
        "speed_limit_used": _smarter_speed_default(default_route["distance_miles"]),
        "route_raw":        route,
        "weather_raw":      weather,
    }

    return features_df, context
