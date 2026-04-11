"""
routes.py — Google Maps Routes API integration.

Calls the Routes API v2 to get traffic-aware driving directions
with alternative routes between two points.

Usage example:
    from src.live.routes import get_route

    # Using address strings
    result = get_route(
        origin="Fenway Park, Boston, MA",
        destination="Boston Logan International Airport, MA"
    )
    print(result["default_route"]["duration_minutes"])

    # Using lat/lng dicts
    result = get_route(
        origin={"lat": 42.3467, "lng": -71.0972},
        destination={"lat": 42.3656, "lng": -71.0096}
    )
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root (two levels up from this file: src/live/routes.py)
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

ROUTES_API_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
FIELD_MASK = (
    "routes.duration,"
    "routes.distanceMeters,"
    "routes.polyline.encodedPolyline,"
    "routes.description,"
    "routes.legs.steps.navigationInstruction"
)


def _build_waypoint(location):
    """
    Accepts either:
      - a dict with 'lat' and 'lng' keys  → lat/lng waypoint
      - a string                           → address waypoint
    """
    if isinstance(location, dict):
        return {
            "location": {
                "latLng": {
                    "latitude": location["lat"],
                    "longitude": location["lng"],
                }
            }
        }
    return {"address": location}


def _parse_route(route):
    """Extract the standard fields from a single route object."""
    duration_seconds = int(route.get("duration", "0s").rstrip("s"))
    distance_meters = route.get("distanceMeters", 0)
    return {
        "duration_seconds": duration_seconds,
        "duration_minutes": round(duration_seconds / 60, 2),
        "distance_meters": distance_meters,
        "distance_km": round(distance_meters / 1000, 3),
        "distance_miles": round(distance_meters / 1609.344, 3),
        "polyline": route.get("polyline", {}).get("encodedPolyline", ""),
    }


def get_route(origin, destination, departure_time=None):
    """
    Get traffic-aware driving directions from origin to destination.

    Args:
        origin (str | dict): Address string OR dict with 'lat' and 'lng' keys.
        destination (str | dict): Address string OR dict with 'lat' and 'lng' keys.
        departure_time (str, optional): ISO 8601 datetime string, e.g.
            "2024-10-15T08:00:00Z". If omitted, Google defaults to now.

    Returns:
        dict: {
            "default_route": {
                "duration_seconds": int,
                "duration_minutes": float,
                "distance_meters": int,
                "distance_km": float,
                "distance_miles": float,
                "polyline": str,
            },
            "alternative_routes": [ { same fields }, ... ],
            "best_alternative_savings_minutes": float,
            "num_alternatives": int,
        }

    Raises:
        EnvironmentError: If GOOGLE_MAPS_API_KEY is not set in .env.
        requests.HTTPError: On non-200 responses from Google.
        ValueError: If the API returns no routes.
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_MAPS_API_KEY not found. "
            f"Make sure it is set in {_ENV_PATH}"
        )

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": FIELD_MASK,
    }

    body = {
        "origin": _build_waypoint(origin),
        "destination": _build_waypoint(destination),
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE_OPTIMAL",
        "computeAlternativeRoutes": True,
        "languageCode": "en-US",
        "units": "IMPERIAL",
    }

    if departure_time:
        body["departureTime"] = departure_time

    try:
        resp = requests.post(ROUTES_API_URL, json=body, headers=headers, timeout=15)
    except requests.ConnectionError as e:
        raise requests.ConnectionError(f"Network error reaching Google Routes API: {e}")

    if not resp.ok:
        try:
            err = resp.json()
            msg = err.get("error", {}).get("message", resp.text)
        except Exception:
            msg = resp.text
        raise requests.HTTPError(
            f"Google Routes API returned HTTP {resp.status_code}: {msg}"
        )

    data = resp.json()
    routes = data.get("routes", [])

    if not routes:
        raise ValueError(
            "Google Routes API returned no routes. "
            "Check that the origin/destination are valid and the API is enabled."
        )

    default = _parse_route(routes[0])
    alternatives = [_parse_route(r) for r in routes[1:]]

    # Best saving = how many minutes faster the quickest alternative is (if any)
    savings = 0.0
    if alternatives:
        fastest_alt = min(alternatives, key=lambda r: r["duration_seconds"])
        diff = default["duration_seconds"] - fastest_alt["duration_seconds"]
        savings = round(max(diff, 0) / 60, 2)

    return {
        "default_route": default,
        "alternative_routes": alternatives,
        "best_alternative_savings_minutes": savings,
        "num_alternatives": len(alternatives),
    }
