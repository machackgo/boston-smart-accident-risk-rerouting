"""
predictor.py — End-to-end route risk prediction.

Ties together the live data integrations (routes, weather) and the v2 model
to produce a complete risk assessment for a driving route.

The model is loaded once at module import time (global singleton) so repeated
calls to predict_route_risk() do not incur repeated disk reads.

Usage:
    from src.predict.predictor import predict_route_risk

    result = predict_route_risk(
        origin="Fenway Park, Boston, MA",
        destination="Boston Logan International Airport, MA",
    )
    print(result["risk_class"])          # e.g. "Medium"
    print(result["confidence"])          # e.g. 0.621
    print(result["class_probabilities"]) # {"High": ..., "Low": ..., "Medium": ...}
"""

import sys
import joblib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from predict.feature_builder import build_features

# ── Load model once at module level ───────────────────────────────────────────
_MODEL_PATH = _REPO_ROOT / "models" / "best_model_v2.pkl"
_model_bundle = joblib.load(_MODEL_PATH)
_MODEL   = _model_bundle["model"]
_CLASSES = _model_bundle["classes"]   # ['High', 'Low', 'Medium']


def predict_route_risk(origin: str, destination: str, departure_time=None) -> dict:
    """
    Predict accident risk severity for a driving route.

    Orchestrates geocoding → routing → weather → feature assembly → model inference
    and returns a structured risk assessment.

    Args:
        origin (str): Starting address or place name.
        destination (str): Destination address or place name.
        departure_time: ISO 8601 string, datetime object, or None (= current time).

    Returns:
        dict: {
            "risk_class": str,               # "Low", "Medium", or "High"
            "confidence": float,             # probability of the predicted class
            "class_probabilities": {
                "Low": float,
                "Medium": float,
                "High": float,
            },
            "route": {
                "duration_minutes": float,
                "distance_miles": float,
                "num_alternatives": int,
                "best_alternative_savings_minutes": float,
            },
            "weather": {
                "condition": str,
                "temperature_f": float,
                "is_precipitation": bool,
                "is_low_visibility": bool,
            },
            "context": {
                "midpoint_lat": float,
                "midpoint_lng": float,
                "hour_of_day": int,
                "weather_mapping_used": str,
            },
        }

    Raises:
        EnvironmentError: If a required API key is missing from .env.
        requests.ConnectionError: On network failures to any live API.
        requests.HTTPError: On non-200 responses from any live API.
        ValueError: If the route returns no polyline or the destination is unreachable.
    """
    # ── Feature assembly (live API calls) ─────────────────────────────────────
    features_df, context = build_features(origin, destination, departure_time)

    # ── Model inference ───────────────────────────────────────────────────────
    risk_class = _MODEL.predict(features_df)[0]
    probas     = _MODEL.predict_proba(features_df)[0]   # order matches _CLASSES

    proba_dict    = {cls: round(float(p), 4) for cls, p in zip(_CLASSES, probas)}
    confidence    = round(float(max(probas)), 4)

    # ── Extract route summary ─────────────────────────────────────────────────
    route_raw    = context["route_raw"]
    default_leg  = route_raw["default_route"]
    weather_raw  = context["weather_raw"]

    return {
        "risk_class": risk_class,
        "confidence": confidence,
        "class_probabilities": proba_dict,
        "route": {
            "duration_minutes":                  default_leg["duration_minutes"],
            "distance_miles":                    default_leg["distance_miles"],
            "num_alternatives":                  route_raw["num_alternatives"],
            "best_alternative_savings_minutes":  route_raw["best_alternative_savings_minutes"],
            "polyline":                          default_leg.get("polyline", ""),
            "alternative_polylines":             [a.get("polyline", "") for a in route_raw["alternative_routes"]],
        },
        "weather": {
            "condition":          weather_raw["condition"],
            "temperature_f":      weather_raw["temperature_f"],
            "is_precipitation":   weather_raw["is_precipitation"],
            "is_low_visibility":  weather_raw["is_low_visibility"],
        },
        "context": {
            "midpoint_lat":         context["midpoint_lat"],
            "midpoint_lng":         context["midpoint_lng"],
            "hour_of_day":          context["hour_of_day"],
            "weather_mapping_used": context["weather_mapping_used"],
        },
    }
