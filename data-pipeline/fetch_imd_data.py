"""
fetch_imd_data.py
------------------
Pulls REAL IMD gridded rainfall + temperature data via the `imdlib` package,
averages it spatially over the state(s) you pick, resamples it to monthly
values, and writes a CSV whose columns are an EXACT match for the
`ClimateRecord` Mongoose schema already in drishyam-backend:

    region, date, year, month, rainfall_mm, temp_c, source

This replaces the synthetic-PRNG part of `npm run seed` for whichever
states you run it for. States you don't run it for will keep using the
synthetic seed data until you get to them — the app doesn't care, since
both write to the same collection with the same shape.

WHY imdlib
----------
IMD (India Meteorological Department) publishes gridded daily rainfall
(0.25° x 0.25°) and daily tmax/tmin (1° x 1°) as binary .grd files, freely,
back to 1901 (rain) / 1951 (temperature). `imdlib` downloads and parses
these for you into xarray objects. This is real IMD data, not a scrape of
a random website — appropriate to cite in an ISRO/IMD hackathon.

REQUIREMENTS
------------
    pip install -r requirements.txt

This script needs to reach IMD's data servers directly, so run it on your
own machine / Colab with normal internet access (not inside a sandboxed
CI environment with an allow-list).

USAGE
-----
    python fetch_imd_data.py --states MH,KL --start 2005 --end 2024

    --states   comma-separated state codes from STATE_BOUNDS below
    --start    first year of data (inclusive)
    --end      last year of data (inclusive)
    --out      output CSV path (default: climate_data.csv)

This can take several minutes on first run — it's downloading ~20 years
of daily gridded data for all of India, then cropping to your state(s).
imdlib caches the raw .grd files under ./imd_raw/ so re-running with a
different --states value for the same year range won't re-download.
"""

import argparse
import os
from datetime import datetime

import pandas as pd
import xarray as xr
import imdlib

# ---------------------------------------------------------------------------
# Approximate bounding boxes (lat_min, lat_max, lon_min, lon_max).
# These are rectangles, not precise state borders, so a rectangle for e.g.
# Maharashtra will pick up a sliver of neighboring states at the grid edges.
# That's an accepted simplification for a hackathon prototype — note it in
# your submission if asked. Add more states here as needed; codes must
# match the `region` ids already used in drishyam-backend/src/utils/regions.js.
# ---------------------------------------------------------------------------
STATE_BOUNDS = {
    "MH": {"name": "Maharashtra", "lat": (15.6, 22.1), "lon": (72.6, 80.9)},
    "KL": {"name": "Kerala", "lat": (8.2, 12.8), "lon": (74.8, 77.4)},
    "TN": {"name": "Tamil Nadu", "lat": (8.1, 13.6), "lon": (76.2, 80.4)},
    "RJ": {"name": "Rajasthan", "lat": (23.0, 30.2), "lon": (69.5, 78.3)},
    "AS": {"name": "Assam", "lat": (24.1, 28.0), "lon": (89.7, 96.1)},
    "WB": {"name": "West Bengal", "lat": (21.5, 27.2), "lon": (85.8, 89.9)},
}

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

RAW_DIR = "imd_raw"


def download_all(start_yr, end_yr):
    """Downloads (or reuses cached) IMD binaries for the whole country."""
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"[fetch] downloading rain {start_yr}-{end_yr} (cached under {RAW_DIR}/rain)...")
    rain = imdlib.get_data("rain", start_yr, end_yr, fn_format="yearwise",
                            file_dir=os.path.join(RAW_DIR, "rain"))
    print(f"[fetch] downloading tmax {start_yr}-{end_yr} (cached under {RAW_DIR}/tmax)...")
    tmax = imdlib.get_data("tmax", start_yr, end_yr, fn_format="yearwise",
                            file_dir=os.path.join(RAW_DIR, "tmax"))
    print(f"[fetch] downloading tmin {start_yr}-{end_yr} (cached under {RAW_DIR}/tmin)...")
    tmin = imdlib.get_data("tmin", start_yr, end_yr, fn_format="yearwise",
                            file_dir=os.path.join(RAW_DIR, "tmin"))
    return rain, tmax, tmin


def crop_to_state(dataset: xr.Dataset, bounds, var_name):
    """
    Crops an xarray Dataset to a state's bounding box and returns the
    spatial mean as a 1D time series (pandas Series indexed by date).
    """
    lat_min, lat_max = bounds["lat"]
    lon_min, lon_max = bounds["lon"]
    cropped = dataset.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    # spatial mean over the remaining grid cells, ignoring the -999 fill
    # value imdlib already masks to NaN in get_xarray()
    series = cropped[var_name].mean(dim=["lat", "lon"], skipna=True).to_pandas()
    return series


def build_state_monthly(state_code, rain_xr, tmax_xr, tmin_xr):
    bounds = STATE_BOUNDS[state_code]

    rain_series = crop_to_state(rain_xr, bounds, "rain")          # daily mm
    tmax_series = crop_to_state(tmax_xr, bounds, "tmax")            # daily °C
    tmin_series = crop_to_state(tmin_xr, bounds, "tmin")            # daily °C

    # Rainfall: MONTHLY TOTAL (matches the "320 mm for Pune, July" style
    # numbers already used across the mock data / schema examples).
    rain_monthly = rain_series.resample("MS").sum()
    # Temperature: monthly mean of the daily mean of tmax/tmin.
    tmax_monthly = tmax_series.resample("MS").mean()
    tmin_monthly = tmin_series.resample("MS").mean()
    temp_monthly = (tmax_monthly + tmin_monthly) / 2

    rows = []
    for ts in rain_monthly.index:
        if ts not in temp_monthly.index:
            continue
        rainfall_val = rain_monthly.loc[ts]
        temp_val = temp_monthly.loc[ts]
        if pd.isna(rainfall_val) or pd.isna(temp_val):
            continue
        rows.append({
            "region": state_code,
            "date": ts.strftime("%Y-%m"),
            "year": ts.year,
            "month": MONTH_NAMES[ts.month - 1],
            "rainfall_mm": round(float(rainfall_val), 1),
            "temp_c": round(float(temp_val), 1),
            "source": "IMD",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Pull real IMD rainfall/temp data for chosen states")
    parser.add_argument("--states", required=True, help="Comma-separated state codes, e.g. MH,KL")
    parser.add_argument("--start", type=int, default=datetime.now().year - 20)
    parser.add_argument("--end", type=int, default=datetime.now().year - 1)
    parser.add_argument("--out", default="climate_data.csv")
    args = parser.parse_args()

    state_codes = [s.strip().upper() for s in args.states.split(",")]
    unknown = [s for s in state_codes if s not in STATE_BOUNDS]
    if unknown:
        raise SystemExit(
            f"Unknown state code(s): {unknown}. "
            f"Add a bounding box for them in STATE_BOUNDS first. "
            f"Known codes: {list(STATE_BOUNDS)}"
        )

    rain, tmax, tmin = download_all(args.start, args.end)
    rain_xr = rain.get_xarray()
    tmax_xr = tmax.get_xarray()
    tmin_xr = tmin.get_xarray()

    all_rows = []
    for code in state_codes:
        print(f"[fetch] aggregating {STATE_BOUNDS[code]['name']} ({code})...")
        rows = build_state_monthly(code, rain_xr, tmax_xr, tmin_xr)
        print(f"[fetch]   -> {len(rows)} monthly records")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows).sort_values(["region", "date"])
    df.to_csv(args.out, index=False)
    print(f"[fetch] wrote {len(df)} rows to {args.out}")
    print("[fetch] next: python import_to_mongo.py --file", args.out)


if __name__ == "__main__":
    main()
