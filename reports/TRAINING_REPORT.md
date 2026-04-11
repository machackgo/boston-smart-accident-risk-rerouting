# Training Report — Boston Smart Accident Risk Rerouting

## v1 Baseline (All Features)

**Date trained:** April 2026
**Script:** `src/model/train.py`
**Data:** `data/crashes_cache.parquet` (47,689 rows)
**Features:** 126 (numeric + time-engineered + 5 one-hot categoricals)

### Model Comparison (v1)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| LightGBM | 0.6556 | **0.4616** |
| RandomForest | 0.7164 | 0.3689 |
| LogisticRegression | 0.5262 | 0.3635 |

**Winner:** LightGBM (macro F1 = 0.4616)

### v1 Feature Groups

- Numeric: `speed_limit`, `numb_vehc`, `ems_hotspot_flag`, `ems_ped_hotspot_flag`,
  `ems_peak_hour`, `district_num`, `lat`, `lon`
- Time-engineered: `hour_of_day`, `day_of_week`, `month`, `is_weekend`, `is_rush_hour`
- One-hot (5 categoricals): `weath_cond_descr`, `road_surf_cond_descr`,
  `ambnt_light_descr`, `manr_coll_descr`, `rdwy_jnct_type_descr`, `city_town_name`

### v1 Known Design Flaw

v1 included post-crash features — manner of collision, road surface condition, ambient
light description, and number of vehicles involved — that are only known *after* a
crash occurs. These features are not available at live prediction time and constitute
data leakage relative to a real-world rerouting use case. The higher F1 for v1 is
partly attributable to this leakage.

---

## v2 Retraining (Forward-Knowable Features Only)

**Date trained:** April 2026
**Script:** `src/model/train_v2.py`
**Data:** `data/crashes_cache.parquet` (same cache, no API re-hit)
**Features:** 80 (all forward-knowable — see full list below)
**Model artifact:** `models/best_model_v2.pkl`

### Design Change

The core motivation for v2 was to eliminate data leakage introduced by post-crash
signals in v1. A rerouting system scores road segments **before** a crash occurs, so
all inference-time features must be knowable in advance. v2 retains only:

- **Location:** `lat`, `lon`
- **Time:** `hour_of_day`, `day_of_week`, `month`, `is_weekend`, `is_rush_hour`
- **Road property:** `speed_limit`
- **Weather:** `weath_cond_descr` one-hot (observable by driver before any crash)
- **Light proxy:** `light_phase` engineered from `hour_of_day` (not post-crash `ambnt_light_descr`)

**Dropped from v1:**
`manr_coll_descr`, `road_surf_cond_descr`, `ambnt_light_descr`, `numb_vehc`,
`rdwy_jnct_type_descr`, `ems_hotspot_flag`, `ems_ped_hotspot_flag`, `ems_peak_hour`,
`district_num`, `city_town_name`

### Row Counts

| Stage | Rows |
|-------|------|
| Raw parquet | 47,689 |
| After dropping missing target | 43,199 |
| After dropping missing lat/lon/datetime | 40,438 |
| Train (80%) | 32,350 |
| Test (20%) | 8,088 |

### Model Comparison (v2)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| LightGBM | 0.6209 | **0.4025** |
| RandomForest | 0.6811 | 0.3412 |
| LogisticRegression | 0.4591 | 0.3253 |

**Winner:** LightGBM (macro F1 = 0.4025)

### v1 vs v2 Macro F1 Comparison

| Version | Macro F1 | Forward-Knowable | Notes |
|---------|----------|-----------------|-------|
| v1 | 0.4616 | No | Includes post-crash leakage |
| v2 | 0.4025 | Yes | Clean — deployable |
| Delta | −0.0591 | — | Expected drop after removing leakage |

The ~6-point drop in macro F1 is expected and correct. Post-crash features in v1
(especially manner of collision) are definitionally correlated with severity, so
removing them reduces apparent performance. v2 reflects genuine signal available
before a crash.

### v2 Per-Class Metrics (LightGBM, test set)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| High (Fatal) | 0.02 | 0.04 | 0.03 | 45 |
| Low (No Injury) | 0.77 | 0.66 | 0.71 | 5,621 |
| Medium (Injury) | 0.41 | 0.54 | 0.47 | 2,422 |
| Macro avg | 0.40 | 0.42 | 0.40 | 8,088 |

### v2 Top 10 Feature Importances (LightGBM split gain)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `lat` | 4,487 |
| 2 | `lon` | 4,440 |
| 3 | `hour_of_day` | 2,196 |
| 4 | `month` | 1,983 |
| 5 | `day_of_week` | 1,448 |
| 6 | `speed_limit` | 1,308 |
| 7 | `weath_cond_descr_Clear` | 320 |
| 8 | `is_rush_hour` | 255 |
| 9 | `weath_cond_descr_Not_Reported` | 201 |
| 10 | `weath_cond_descr_Clear_Clear` | 165 |

Location and time are the dominant signals. Speed limit is a meaningful road-level
predictor. Weather contributes at the margin.

### v2 Full Feature List (80 features)

```
lat, lon, speed_limit, hour_of_day, day_of_week, month, is_weekend, is_rush_hour,
weath_cond_descr_Blowing_sand__snow, weath_cond_descr_Blowing_sand__snow_Blowing_sand__snow,
weath_cond_descr_Clear, weath_cond_descr_Clear_Blowing_sand__snow,
weath_cond_descr_Clear_Clear, weath_cond_descr_Clear_Cloudy,
weath_cond_descr_Clear_Fog__smog__smoke, weath_cond_descr_Clear_Other,
weath_cond_descr_Clear_Rain, weath_cond_descr_Clear_Severe_crosswinds,
weath_cond_descr_Clear_Sleet__hail__freezing_rain_or_drizzle_,
weath_cond_descr_Clear_Snow, weath_cond_descr_Clear_Unknown,
weath_cond_descr_Cloudy, weath_cond_descr_Cloudy_Clear, weath_cond_descr_Cloudy_Cloudy,
weath_cond_descr_Cloudy_Fog__smog__smoke, weath_cond_descr_Cloudy_Other,
weath_cond_descr_Cloudy_Rain, weath_cond_descr_Cloudy_Severe_crosswinds,
weath_cond_descr_Cloudy_Sleet__hail__freezing_rain_or_drizzle_,
weath_cond_descr_Cloudy_Snow, weath_cond_descr_Cloudy_Unknown,
weath_cond_descr_Fog__smog__smoke, weath_cond_descr_Fog__smog__smoke_Fog__smog__smoke,
weath_cond_descr_Fog__smog__smoke_Rain, weath_cond_descr_Not_Reported,
weath_cond_descr_Other, weath_cond_descr_Other_Other, weath_cond_descr_Other_Rain,
weath_cond_descr_Other_Snow, weath_cond_descr_Other_Unknown,
weath_cond_descr_Rain, weath_cond_descr_Rain_Clear, weath_cond_descr_Rain_Cloudy,
weath_cond_descr_Rain_Fog__smog__smoke, weath_cond_descr_Rain_Other,
weath_cond_descr_Rain_Rain, weath_cond_descr_Rain_Severe_crosswinds,
weath_cond_descr_Rain_Sleet__hail__freezing_rain_or_drizzle_,
weath_cond_descr_Rain_Snow, weath_cond_descr_Rain_Unknown,
weath_cond_descr_Reported_but_invalid, weath_cond_descr_Severe_crosswinds,
weath_cond_descr_Severe_crosswinds_Blowing_sand__snow,
weath_cond_descr_Severe_crosswinds_Clear, weath_cond_descr_Severe_crosswinds_Other,
weath_cond_descr_Severe_crosswinds_Rain,
weath_cond_descr_Severe_crosswinds_Severe_crosswinds,
weath_cond_descr_Sleet__hail__freezing_rain_or_drizzle_,
weath_cond_descr_Sleet__hail__freezing_rain_or_drizzle__Blowing_sand__snow,
weath_cond_descr_Sleet__hail__freezing_rain_or_drizzle__Cloudy,
weath_cond_descr_Sleet__hail__freezing_rain_or_drizzle__Rain,
weath_cond_descr_Sleet__hail__freezing_rain_or_drizzle__Severe_crosswinds,
weath_cond_descr_Sleet__hail__freezing_rain_or_drizzle__Sleet__hail__freezing_rain_or_drizzle_,
weath_cond_descr_Sleet__hail__freezing_rain_or_drizzle__Snow,
weath_cond_descr_Snow, weath_cond_descr_Snow_Blowing_sand__snow,
weath_cond_descr_Snow_Clear, weath_cond_descr_Snow_Cloudy,
weath_cond_descr_Snow_Fog__smog__smoke, weath_cond_descr_Snow_Other,
weath_cond_descr_Snow_Rain, weath_cond_descr_Snow_Severe_crosswinds,
weath_cond_descr_Snow_Sleet__hail__freezing_rain_or_drizzle_,
weath_cond_descr_Snow_Snow, weath_cond_descr_Unknown,
weath_cond_descr_Unknown_Clear, weath_cond_descr_Unknown_Unknown,
light_phase_Dark, light_phase_Dawn_Dusk, light_phase_Daylight
```

### Known Limitations

- **Fatal class barely detected** (F1 = 0.03, support = 45 in test set). The 0.6%
  base rate makes this class extremely hard to learn without oversampling or a
  dedicated anomaly detector.
- **Random train/test split** — temporal leakage not fully addressed. Future work
  should use a date-based split (train on earlier years, test on later years).
- **Speed limit imputed** for ~22% of records. A geocoded lookup from OSM would be
  more accurate.
- **No road-network features** (street type, intersection count, AADT traffic volume).
  Adding these would likely close much of the v1 → v2 performance gap without
  reintroducing data leakage.
