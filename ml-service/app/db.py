"""
db.py
-----
Thin MongoDB access layer. Reads the exact same `climaterecords`
collection the Express backend writes to — this service never needs its
own copy of the data, and never writes back to Mongo itself (Node owns
writing Forecast/Scenario/RiskReport documents).
"""

import os
from functools import lru_cache

from pymongo import MongoClient

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@lru_cache
def get_db():
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is not set. Copy .env.example to .env and fill it in.")
    client = MongoClient(uri)
    return client.get_default_database()


def get_region_history(region: str):
    """
    Returns the region's full monthly history sorted by date ascending, as
    a list of dicts with at least: date, year, month, rainfall_mm, temp_c.
    """
    db = get_db()
    docs = list(
        db.climaterecords.find({"region": region.upper()}, {"_id": 0})
        .sort("date", 1)
    )
    return docs


def historical_mean_for_month(history, month_name: str):
    vals = [d["rainfall_mm"] for d in history if d["month"] == month_name]
    if not vals:
        return None
    return sum(vals) / len(vals)
