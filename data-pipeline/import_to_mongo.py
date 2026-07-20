"""
import_to_mongo.py
-------------------
Loads the CSV produced by fetch_imd_data.py into the SAME MongoDB
collection drishyam-backend already reads from (`climaterecords`),
using the exact field names of the ClimateRecord Mongoose schema:

    region, district, date, year, month, rainfall_mm, temp_c, source

Upserts on (region, date) — the same compound unique index already
defined in ClimateRecord.js — so re-running this script is safe and
just overwrites old values with fresh ones instead of creating
duplicates or crashing on a duplicate-key error.

USAGE
-----
    pip install -r requirements.txt
    cp .env.example .env   # point MONGODB_URI at the same DB the backend uses
    python import_to_mongo.py --file climate_data.csv

After this runs, GET /api/climate/MH/history on your Express backend
should return real IMD numbers instead of synthetic ones — no backend
code changes needed, because the shape matches exactly what
climate.controller.js already expects.
"""

import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Import cleaned IMD CSV into MongoDB")
    parser.add_argument("--file", required=True, help="CSV produced by fetch_imd_data.py")
    parser.add_argument("--collection", default="climaterecords",
                         help="Mongoose pluralizes 'ClimateRecord' to this by default")
    args = parser.parse_args()

    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise SystemExit("MONGODB_URI is not set. Copy .env.example to .env and fill it in.")

    df = pd.read_csv(args.file)
    required_cols = {"region", "date", "year", "month", "rainfall_mm", "temp_c", "source"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing required columns: {missing}")

    client = MongoClient(uri)
    db = client.get_default_database()
    coll = db[args.collection]

    ops = []
    for _, row in df.iterrows():
        doc = {
            "region": row["region"],
            "district": None,
            "date": row["date"],
            "year": int(row["year"]),
            "month": row["month"],
            "rainfall_mm": float(row["rainfall_mm"]),
            "temp_c": float(row["temp_c"]),
            "source": row["source"],
        }
        ops.append(
            UpdateOne(
                {"region": doc["region"], "date": doc["date"]},
                {"$set": doc},
                upsert=True,
            )
        )

    if not ops:
        print("[import] nothing to import — CSV was empty")
        return

    result = coll.bulk_write(ops, ordered=False)
    print(
        f"[import] upserted into '{args.collection}': "
        f"{result.upserted_count} new, {result.modified_count} updated"
    )
    regions = sorted(df["region"].unique())
    print(f"[import] regions touched: {regions}")
    print("[import] done. Restart/re-hit your backend — it now serves real IMD data for these regions.")


if __name__ == "__main__":
    main()
