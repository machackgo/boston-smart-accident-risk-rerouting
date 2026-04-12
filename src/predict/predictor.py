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
import pandas as pd
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from predict.feature_builder import build_features, build_segment_features
from live.routes import get_route
from live.geocoding import reverse_geocode

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


def _sample_leg(leg: dict, num_segments: int) -> tuple[list, list]:
    """
    Return (indices, sample_points) for evenly-spaced samples along a decoded polyline.

    Always includes the first and last point.  Returns ([], []) if the leg has
    fewer than 2 decoded points.
    """
    pts   = leg["decoded_points"]
    total = len(pts)
    if total < 2:
        return [], []
    n    = min(num_segments, total)
    step = max(1, total // n)
    idx  = list(range(0, total, step))[:n]
    if idx[-1] != total - 1:
        idx[-1] = total - 1
    return idx, [(pts[i][0], pts[i][1]) for i in idx]


def _build_recommendation_reason(route_results: list) -> str:
    """Generate a short human-readable sentence explaining the recommendation."""
    if len(route_results) == 1:
        return "Only one route available — no comparison possible."

    rec      = next(r for r in route_results if r["index"] == min(
        route_results, key=lambda r: r["safety_score"])["index"])
    default  = route_results[0]

    if rec["index"] == 0:
        if rec["num_hotspots"] == 0:
            return "Default route is already the safest: no elevated-risk hotspots detected."
        return (
            f"Default route is the safest available with "
            f"{rec['num_hotspots']} hotspot(s) "
            f"({rec['num_high_hotspots']} high-risk)."
        )

    # rec is an alternative
    rec_high = rec["num_high_hotspots"]
    def_high = default["num_high_hotspots"]
    time_diff = round(rec["duration_minutes"] - default["duration_minutes"], 1)

    if rec_high < def_high:
        risk_str = (
            f"{def_high - rec_high} fewer high-risk hotspot(s) vs default route"
        )
    elif rec["num_hotspots"] < default["num_hotspots"]:
        risk_str = (
            f"{default['num_hotspots'] - rec['num_hotspots']} fewer total hotspot(s) vs default route"
        )
    else:
        risk_str = (
            f"lower safety score ({rec['safety_score']:.1f} vs {default['safety_score']:.1f})"
        )

    if abs(time_diff) < 0.5:
        time_str = "at the same travel time"
    elif time_diff > 0:
        time_str = f"at cost of {abs(time_diff):.1f} extra min"
    else:
        time_str = f"saving {abs(time_diff):.1f} min"

    return f"{rec['label']} is safest: {risk_str}, {time_str}."


def predict_route_risk_segmented(
    origin: str,
    destination: str,
    departure_time=None,
    num_segments: int = 12,
) -> dict:
    """
    Predict accident risk across ALL routes (default + alternatives) using a
    single batch LightGBM inference call.

    For each route:
      - Samples num_segments points evenly along the decoded polyline.
      - All routes' feature rows are stacked into ONE DataFrame — one model call
        for all N_routes × num_segments rows (fast even for 3 routes × 12 pts).
      - Weather is fetched ONCE at the default route's midpoint.
      - Hotspots (Medium or High segments) are reverse-geocoded for street names.
      - A safety_score is computed: (high_hotspots×10) + (med_hotspots×3) + (dur×0.1).
      - The route with the lowest safety_score is recommended.

    Args:
        origin (str): Starting address or place name.
        destination (str): Destination address or place name.
        departure_time: ISO 8601 string, datetime object, or None (= current time).
        num_segments (int): Sample points per route (default 12).

    Returns:
        dict with keys: recommended_route_index, default_route_index, routes[],
        weather, context, recommendation_reason.

    NOTE: Response shape changed from the single-route version.
    The old shape (risk_class, segments, hotspots, route at top-level) has been
    replaced by routes[] containing per-route objects.
    """
    # ── 1. Fetch all routes ───────────────────────────────────────────────────
    print(f"[predictor] Fetching routes: '{origin}' → '{destination}' ...")
    route_data = get_route(origin, destination)
    all_legs   = [route_data["default_route"]] + route_data["alternative_routes"]

    # ── 2. Sample points from each leg ───────────────────────────────────────
    leg_meta        = []
    all_sample_pts  = []
    row_offset      = 0

    for leg_idx, leg in enumerate(all_legs):
        indices, sample_pts = _sample_leg(leg, num_segments)
        if not indices:
            print(f"[predictor] Skipping route {leg_idx}: too few polyline points.")
            continue
        leg_meta.append({
            "leg_idx":    leg_idx,
            "leg":        leg,
            "indices":    indices,
            "sample_pts": sample_pts,
            "total_pts":  len(leg["decoded_points"]),
            "row_start":  row_offset,
            "num_rows":   len(indices),
        })
        all_sample_pts.extend(sample_pts)
        row_offset += len(indices)

    if not leg_meta:
        raise ValueError("No valid routes with enough decoded polyline points.")

    print(f"[predictor] Analyzing {len(leg_meta)} route(s), "
          f"{len(all_sample_pts)} total sample points.")

    # ── 3. Build ALL feature rows — ONE weather call ──────────────────────────
    all_features_df, feat_ctx = build_segment_features(
        route_data, all_sample_pts, departure_time
    )

    # ── 4. Single batch inference ─────────────────────────────────────────────
    all_predictions = _MODEL.predict(all_features_df)
    all_probas      = _MODEL.predict_proba(all_features_df)

    # ── 5. Assemble per-route results ─────────────────────────────────────────
    weather_raw   = feat_ctx["weather_raw"]
    route_results = []

    for meta in leg_meta:
        leg        = meta["leg"]
        start, end = meta["row_start"], meta["row_start"] + meta["num_rows"]
        preds      = all_predictions[start:end]
        probas     = all_probas[start:end]
        indices    = meta["indices"]
        sample_pts = meta["sample_pts"]
        total_pts  = meta["total_pts"]
        dist_miles = leg["distance_miles"]
        dur_min    = leg["duration_minutes"]

        # ── Segments ─────────────────────────────────────────────────────────
        segments = []
        for i, (poly_idx, (lat, lng), risk_class, proba) in enumerate(
            zip(indices, sample_pts, preds, probas)
        ):
            pd_dict    = {cls: round(float(p), 4) for cls, p in zip(_CLASSES, proba)}
            confidence = round(float(max(proba)), 4)
            dist_start = round((poly_idx / (total_pts - 1)) * dist_miles, 3)
            segments.append({
                "index":                     i,
                "polyline_index":            poly_idx,
                "lat":                       lat,
                "lng":                       lng,
                "distance_from_start_miles": dist_start,
                "risk_class":                risk_class,
                "confidence":                confidence,
                "probabilities":             pd_dict,
            })

        # ── Hotspots (Medium/High) — reverse-geocode each ─────────────────────
        hotspots = []
        for s in segments:
            if _RISK_ORDER.get(s["risk_class"], 0) < 1:
                continue
            est_min = round(
                (s["distance_from_start_miles"] / dist_miles) * dur_min, 1
            ) if dist_miles > 0 else 0.0
            geo_info = {
                "street_name": None, "neighborhood": None, "short_label": None,
            }
            try:
                geo = reverse_geocode(s["lat"], s["lng"])
                geo_info = {
                    "street_name": geo.get("street_name"),
                    "neighborhood": geo.get("neighborhood"),
                    "short_label": geo.get("short_label"),
                }
            except Exception as e:
                print(f"[predictor] reverse_geocode failed for ({s['lat']}, {s['lng']}): {e}")
            hotspots.append({
                "index":                     s["index"],
                "lat":                       s["lat"],
                "lng":                       s["lng"],
                "distance_from_start_miles": s["distance_from_start_miles"],
                "estimated_minutes_from_start": est_min,
                "risk_class":                s["risk_class"],
                "confidence":                s["confidence"],
                **geo_info,
            })

        # ── Overall risk & avg probabilities ─────────────────────────────────
        worst_seg    = max(segments, key=lambda s: _RISK_ORDER.get(s["risk_class"], 0))
        overall_risk = worst_seg["risk_class"]
        avg_proba    = {
            cls: round(sum(s["probabilities"][cls] for s in segments) / len(segments), 4)
            for cls in _CLASSES
        }
        overall_conf = round(avg_proba[overall_risk], 4)

        # ── Safety score ──────────────────────────────────────────────────────
        n_high   = sum(1 for h in hotspots if h["risk_class"] == "High")
        n_medium = sum(1 for h in hotspots if h["risk_class"] == "Medium")
        safety   = round((n_high * 10) + (n_medium * 3) + (dur_min * 0.1), 3)

        is_default = (meta["leg_idx"] == 0)
        label      = "Default route" if is_default else f"Alternative {meta['leg_idx']}"

        route_results.append({
            "index":                 meta["leg_idx"],
            "is_default":            is_default,
            "label":                 label,
            "duration_minutes":      dur_min,
            "distance_miles":        dist_miles,
            "safety_score":          safety,
            "num_hotspots":          len(hotspots),
            "num_high_hotspots":     n_high,
            "num_medium_hotspots":   n_medium,
            "overall_risk_class":    overall_risk,
            "overall_confidence":    overall_conf,
            "overall_probabilities": avg_proba,
            "segments":              segments,
            "hotspots":              hotspots,
            "decoded_points":        leg["decoded_points"],
            "polyline":              leg.get("polyline", ""),
        })

    # ── 6. Pick recommended route (lowest safety score) ──────────────────────
    best              = min(route_results, key=lambda r: r["safety_score"])
    rec_route_idx     = best["index"]
    reason            = _build_recommendation_reason(route_results)

    print(f"[predictor] Recommended: route {rec_route_idx} ({best['label']}) "
          f"safety_score={best['safety_score']}")

    return {
        "recommended_route_index": rec_route_idx,
        "default_route_index":     0,
        "routes":                  route_results,
        "weather": {
            "condition":         weather_raw["condition"],
            "temperature_f":     weather_raw["temperature_f"],
            "is_precipitation":  weather_raw["is_precipitation"],
            "is_low_visibility": weather_raw["is_low_visibility"],
        },
        "context": {
            "timezone":              "America/New_York",
            "local_hour":            feat_ctx["local_hour"],
            "num_routes_analyzed":   len(route_results),
            "num_segments_per_route": num_segments,
            "weather_mapping_used":  feat_ctx["weather_mapping_used"],
        },
        "recommendation_reason": reason,
    }
