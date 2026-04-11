"""
geocoding.py — Google Maps Geocoding API integration.

Converts a human-readable address into geographic coordinates (lat/lng)
using the Google Maps Geocoding API.

Usage example:
    from src.live.geocoding import geocode

    result = geocode("Fenway Park, Boston, MA")
    print(result["lat"])               # 42.3467...
    print(result["lng"])               # -71.0972...
    print(result["formatted_address"]) # "4 Jersey St, Boston, MA 02215, USA"
    print(result["location_type"])     # "GEOMETRIC_CENTER"
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root (two levels up from this file: src/live/geocoding.py)
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

GEOCODING_API_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def geocode(address: str) -> dict:
    """
    Convert a human-readable address to geographic coordinates.

    Args:
        address (str): The address or place name to geocode.

    Returns:
        dict: {
            "lat": float,             # latitude
            "lng": float,             # longitude
            "formatted_address": str, # Google's canonical address string
            "place_id": str,          # Google's unique place identifier
            "location_type": str,     # accuracy: "ROOFTOP", "RANGE_INTERPOLATED",
                                      #           "GEOMETRIC_CENTER", or "APPROXIMATE"
        }

    Raises:
        EnvironmentError: If GOOGLE_MAPS_API_KEY is not set in .env.
        requests.ConnectionError: On network errors reaching Google.
        requests.HTTPError: On non-200 HTTP responses from Google.
        ValueError: If Google returns ZERO_RESULTS or a non-OK status.

    Example:
        # Geocode a Boston landmark
        result = geocode("Fenway Park, Boston, MA")
        print(result["lat"], result["lng"])
        # → 42.3466764  -71.09715390000001

        # Use coordinates directly in a downstream call
        from src.live.weather import get_weather
        weather = get_weather(lat=result["lat"], lng=result["lng"])
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_MAPS_API_KEY not found. "
            f"Make sure it is set in {_ENV_PATH}"
        )

    params = {
        "address": address,
        "key": api_key,
    }

    try:
        resp = requests.get(GEOCODING_API_URL, params=params, timeout=15)
    except requests.ConnectionError as e:
        raise requests.ConnectionError(f"Network error reaching Google Geocoding API: {e}")

    if not resp.ok:
        try:
            err = resp.json()
            msg = err.get("error_message", resp.text)
        except Exception:
            msg = resp.text
        raise requests.HTTPError(
            f"Google Geocoding API returned HTTP {resp.status_code}: {msg}"
        )

    data = resp.json()
    status = data.get("status")

    if status == "ZERO_RESULTS":
        raise ValueError(
            f"Google Geocoding API returned no results for address: '{address}'. "
            "Check the address is valid and spelled correctly."
        )

    if status != "OK":
        error_message = data.get("error_message", "No error message provided.")
        raise ValueError(
            f"Google Geocoding API returned status '{status}' for address: '{address}'. "
            f"Details: {error_message}"
        )

    result = data["results"][0]
    location = result["geometry"]["location"]

    return {
        "lat": float(location["lat"]),
        "lng": float(location["lng"]),
        "formatted_address": result["formatted_address"],
        "place_id": result["place_id"],
        "location_type": result["geometry"]["location_type"],
    }
