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

from predict.feature_builder import build_features, build_segment_features
from live.routes import get_route

# ── Load model once at module level ───────────────────────────────────────────
_MODEL_PATH = _REPO_ROOT / "models" / "best_model_v2.pkl"
_model_bundle = joblib.load(_MODEL_PATH)
_MODEL   = _model_bundle["model"]
_CLASSES = _model_bundle["classes"]   # ['High', 'Low', 'Medium']

# Severity ranking for worst-case aggregation
_RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2}


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


def predict_route_risk_segmented(
    origin: str,
    destination: str,
    departure_time=None,
    num_segments: int = 12,
) -> dict:
    """
    Predict accident risk at multiple points along a route using batch inference.

    Samples num_segments points evenly from the decoded polyline, runs a single
    batch predict/predict_proba call (LightGBM handles it in microseconds), and
    returns per-segment risk classes alongside an overall worst-case assessment.

    Args:
        origin (str): Starting address or place name.
        destination (str): Destination address or place name.
        departure_time: ISO 8601 string, datetime object, or None (= current time).
        num_segments (int): Number of sample points along the route (default 12).

    Returns:
        dict with keys: risk_class, overall_confidence, overall_probabilities,
        segments, hotspots, route (with decoded_points), weather, context.
    """
    # ── Route (geocodes addresses, decodes polyline) ──────────────────────────
    route        = get_route(origin, destination)
    default_leg  = route["default_route"]
    decoded_pts  = default_leg["decoded_points"]   # list of (lat, lng) tuples
    total_pts    = len(decoded_pts)

    if total_pts < 2:
        raise ValueError(
            "Route returned fewer than 2 decoded polyline points — "
            "cannot perform segmented analysis."
        )

    # ── Sample points evenly; always include first and last ───────────────────
    n    = min(num_segments, total_pts)
    step = max(1, total_pts // n)
    indices = list(range(0, total_pts, step))[:n]
    # Guarantee we end exactly at the last point
    if indices[-1] != total_pts - 1:
        indices[-1] = total_pts - 1

    sample_points = [(decoded_pts[i][0], decoded_pts[i][1]) for i in indices]

    # ── Build features + one weather call ────────────────────────────────────
    features_df, feat_ctx = build_segment_features(route, sample_points, departure_time)

    # ── Batch model inference ─────────────────────────────────────────────────
    predictions   = _MODEL.predict(features_df)           # shape (n,)
    proba_matrix  = _MODEL.predict_proba(features_df)     # shape (n, 3)

    # ── Assemble segments ─────────────────────────────────────────────────────
    distance_miles = default_leg["distance_miles"]
    segments = []
    for i, (poly_idx, (lat, lng), risk_class, probas) in enumerate(
        zip(indices, sample_points, predictions, proba_matrix)
    ):
        proba_dict = {cls: round(float(p), 4) for cls, p in zip(_CLASSES, probas)}
        confidence = round(float(max(probas)), 4)
        dist_from_start = round(
            (poly_idx / (total_pts - 1)) * distance_miles, 3
        )
        segments.append({
            "index":                   i,
            "polyline_index":          poly_idx,
            "lat":                     lat,
            "lng":                     lng,
            "distance_from_start_miles": dist_from_start,
            "risk_class":              risk_class,
            "confidence":              confidence,
            "probabilities":           proba_dict,
        })

    # ── Hotspots: any segment classified Medium or High ───────────────────────
    hotspots = [
        {
            "index":                   s["index"],
            "lat":                     s["lat"],
            "lng":                     s["lng"],
            "distance_from_start_miles": s["distance_from_start_miles"],
            "risk_class":              s["risk_class"],
            "confidence":              s["confidence"],
        }
        for s in segments
        if _RISK_ORDER.get(s["risk_class"], 0) >= 1
    ]

    # ── Overall risk: worst segment (safety-conservative) ────────────────────
    worst_seg    = max(segments, key=lambda s: _RISK_ORDER.get(s["risk_class"], 0))
    overall_risk = worst_seg["risk_class"]

    # Average probabilities across all segments
    avg_proba = {
        cls: round(sum(s["probabilities"][cls] for s in segments) / len(segments), 4)
        for cls in _CLASSES
    }
    overall_confidence = round(avg_proba[overall_risk], 4)

    weather_raw = feat_ctx["weather_raw"]

    return {
        "risk_class":            overall_risk,
        "overall_confidence":    overall_confidence,
        "overall_probabilities": avg_proba,
        "segments":              segments,
        "hotspots":              hotspots,
        "route": {
            "duration_minutes":                 default_leg["duration_minutes"],
            "distance_miles":                   default_leg["distance_miles"],
            "num_alternatives":                 route["num_alternatives"],
            "best_alternative_savings_minutes": route["best_alternative_savings_minutes"],
            "polyline":                         default_leg.get("polyline", ""),
            "decoded_points":                   decoded_pts,
            "alternative_polylines":            [a.get("polyline", "") for a in route["alternative_routes"]],
        },
        "weather": {
            "condition":         weather_raw["condition"],
            "temperature_f":     weather_raw["temperature_f"],
            "is_precipitation":  weather_raw["is_precipitation"],
            "is_low_visibility": weather_raw["is_low_visibility"],
        },
        "context": {
            "timezone":              "America/New_York",
            "local_hour":            feat_ctx["local_hour"],
            "num_segments_analyzed": len(segments),
            "num_hotspots":          len(hotspots),
            "weather_mapping_used":  feat_ctx["weather_mapping_used"],
        },
    }
