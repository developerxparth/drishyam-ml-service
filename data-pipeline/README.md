# Drishyam — Real IMD Data Pipeline

Pulls real IMD gridded rainfall + temperature data (via `imdlib`) for the
state(s) you choose, aggregates it to the same monthly shape your backend
already expects, and loads it straight into the `climaterecords` collection
in MongoDB — replacing the synthetic seed data for those states, with zero
backend code changes.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # point MONGODB_URI at the SAME db drishyam-backend uses
```

## Run

```bash
# Step 1: download + aggregate (needs real internet access, takes a few minutes)
python fetch_imd_data.py --states MH,KL --start 2005 --end 2024

# Step 2: load into MongoDB
python import_to_mongo.py --file climate_data.csv
```

That's it — restart your Express backend (or just hit it again) and
`GET /api/climate/MH/history` now returns real IMD numbers.

## Notes

- `STATE_BOUNDS` in `fetch_imd_data.py` has bounding boxes for MH, KL, TN,
  RJ, AS, WB. Add more by copying the pattern — codes must match the
  `region` ids in `drishyam-backend/src/utils/regions.js`.
- Bounding boxes are rectangles, not precise borders, so there's minor
  spillover into neighboring states at the edges. That's a standard,
  disclosable simplification for an 8-week prototype.
- Rainfall is aggregated as a **monthly total** (not a daily mean), matching
  the numbers already used across your schema/mock data. Temperature is the
  monthly mean of daily (tmax+tmin)/2.
- Re-running `import_to_mongo.py` is always safe — it upserts on
  `(region, date)`, the same unique index already defined on
  `ClimateRecord`, so it never creates duplicates.
- States you haven't run this for keep using the synthetic seed data from
  `npm run seed` in the backend — the two coexist fine since they write to
  the same collection with the same shape.
