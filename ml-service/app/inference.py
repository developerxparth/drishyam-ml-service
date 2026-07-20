"""
inference.py
------------
Loads whichever regions have been trained (models/{region}_lstm.pt +
{region}_meta.json) and serves forecasts from them.

DESIGN NOTE on "month"
-----------------------
The LSTM is a simple one-step-ahead model: given the most recent 12
months on record for a region, it predicts the month immediately after
them. It does not take an arbitrary target month as an input feature.
For this prototype, we treat that one-step prediction as "the forecast
for whatever month Node asks about" (which in practice is the current
month, since that's all `mlService.js` ever requests) and pair it with
that month's historical mean for the anomaly calculation. This is a
reasonable simplification to disclose in your submission — a fuller
version would retrain relative to a rolling window ending the month
before the target, or use a seq2seq/multi-step model.

DESIGN NOTE on scenarios
------------------------
`tempDelta` is not a feature the model was trained on (this is a
univariate rainfall LSTM, matching the "small, explainable, trains in
minutes" scope from the original plan). For `rainfallDelta`, we do
something more genuine than the Node fallback: we scale the actual
input window fed to the model, so the LSTM itself reacts to the
hypothetical wetter/drier recent history. `tempDelta` is then applied
as a physically-motivated multiplier on top (higher temperature ->
more evapotranspiration -> lower effective rainfall benefit), the same
relationship used in the Node fallback, so behavior is consistent
whichever path serves a given request.
"""

import json
import os
from datetime import datetime

import torch

from app.model import load_model, normalize, denormalize, SEQ_LEN
from app.db import historical_mean_for_month, get_region_history, MONTH_NAMES

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def current_month_name() -> str:
    """Mirrors currentTargetMonth() in the Node backend's climateStats.js."""
    return MONTH_NAMES[datetime.now().month - 1]

_cache = {}  # region -> {"model": ..., "meta": {...}}


def available_regions():
    if not os.path.isdir(MODELS_DIR):
        return []
    return sorted(
        f[: -len("_meta.json")]
        for f in os.listdir(MODELS_DIR)
        if f.endswith("_meta.json")
    )


def _load_region(region: str):
    region = region.upper()
    if region in _cache:
        return _cache[region]

    meta_path = os.path.join(MODELS_DIR, f"{region}_meta.json")
    model_path = os.path.join(MODELS_DIR, f"{region}_lstm.pt")
    if not (os.path.exists(meta_path) and os.path.exists(model_path)):
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    model = load_model(model_path, hidden_size=meta.get("hidden_size", 32))

    entry = {"model": model, "meta": meta}
    _cache[region] = entry
    return entry


def _run_model(model, window, vmin, vmax):
    norm_window = normalize(window, vmin, vmax)
    x = torch.tensor(norm_window, dtype=torch.float32).view(1, SEQ_LEN, 1)
    with torch.no_grad():
        pred_norm = model(x).item()
    return denormalize(pred_norm, vmin, vmax)


def predict(region: str, month: str):
    """Returns None if this region hasn't been trained yet (caller should 404)."""
    entry = _load_region(region)
    if entry is None:
        return None

    model, meta = entry["model"], entry["meta"]
    window = meta["last_window"]
    predicted = _run_model(model, window, meta["min"], meta["max"])
    predicted = max(0.0, predicted)

    hist_mean = meta["monthly_means"].get(month)
    if hist_mean is None:
        # fall back to live Mongo lookup in case meta predates new data
        history = get_region_history(region)
        hist_mean = historical_mean_for_month(history, month) or predicted

    anomaly_pct = round(((predicted - hist_mean) / hist_mean) * 100) if hist_mean else 0

    return {
        "month": month,
        "predicted_rainfall_mm": round(predicted),
        "historical_mean_mm": round(hist_mean),
        "anomaly_pct": anomaly_pct,
    }


def run_scenario(region: str, rainfall_delta: float, temp_delta: float):
    """
    Note: unlike /predict, Node's /api/scenario does NOT send a `month`
    field (see mlService.js: callMlService('/scenario', { region, ...inputs })).
    We compute "the current month" the same way the Node side does
    (climateStats.js's currentTargetMonth), so both paths agree on which
    month a scenario result is labeled with.
    """
    month = current_month_name()
    entry = _load_region(region)
    if entry is None:
        return None

    model, meta = entry["model"], entry["meta"]
    window = meta["last_window"]

    # Scale the actual input window fed to the model — the LSTM reacts to
    # a hypothetically wetter/drier recent history, not just a post-hoc
    # multiplier on its output.
    rainfall_factor = 1 + (rainfall_delta / 100)
    scaled_window = [v * rainfall_factor for v in window]

    predicted = _run_model(model, scaled_window, meta["min"], meta["max"])

    # Temperature effect: not a model input, applied as a physically
    # motivated multiplier (higher temp -> more evapotranspiration ->
    # less effective rainfall), matching the Node fallback's formula.
    temp_penalty = 1 - max(0, temp_delta) * 0.025
    predicted = max(0.0, predicted * temp_penalty)

    hist_mean = meta["monthly_means"].get(month, predicted)
    anomaly_pct = round(((predicted - hist_mean) / hist_mean) * 100) if hist_mean else 0

    return {
        "month": month,
        "predicted_rainfall_mm": round(predicted),
        "historical_mean_mm": round(hist_mean),
        "anomaly_pct": anomaly_pct,
    }
