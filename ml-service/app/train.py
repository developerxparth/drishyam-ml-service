"""
train.py
--------
Trains one LSTM per region, reading history straight from the same
MongoDB the Express backend uses (climaterecords). Run this offline,
once per region, after you've loaded real IMD data with the
data-pipeline scripts.

USAGE
-----
    cd ml-service
    pip install -r requirements.txt
    cp .env.example .env        # same MONGODB_URI as the backend
    python -m app.train --region MH
    python -m app.train --region KL

Produces, per region, under ./models/:
    {region}_lstm.pt    - PyTorch state dict
    {region}_meta.json  - normalization range, per-month historical means,
                          and the most recent 12-month window (needed at
                          inference time to actually forecast "next month")

These two files are everything main.py needs to serve /predict for that
region — no Mongo lookups needed at request time for the model itself
(though /scenario still reads history for consistency checks).
"""

import argparse
import json
import os

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn

from app.db import get_region_history, MONTH_NAMES
from app.model import RainfallLSTM, make_sequences, normalize, SEQ_LEN

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train_region(region: str, epochs: int = 300, lr: float = 0.01):
    history = get_region_history(region)
    if len(history) < SEQ_LEN + 12:
        raise SystemExit(
            f"Not enough history for {region}: {len(history)} months found, "
            f"need at least {SEQ_LEN + 12}. Run the data-pipeline import first."
        )

    values = [d["rainfall_mm"] for d in history]
    vmin, vmax = min(values), max(values)
    normalized = normalize(values, vmin, vmax)

    X, y = make_sequences(normalized, SEQ_LEN)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)   # (N, SEQ_LEN, 1)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)   # (N, 1)

    model = RainfallLSTM(hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"[train:{region}] epoch {epoch + 1}/{epochs}  loss={loss.item():.5f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{region}_lstm.pt")
    torch.save(model.state_dict(), model_path)

    monthly_means = {}
    for m in MONTH_NAMES:
        vals = [d["rainfall_mm"] for d in history if d["month"] == m]
        if vals:
            monthly_means[m] = round(sum(vals) / len(vals), 2)

    meta = {
        "region": region,
        "min": vmin,
        "max": vmax,
        "seq_len": SEQ_LEN,
        "monthly_means": monthly_means,
        "last_window": values[-SEQ_LEN:],   # most recent 12 months, raw mm
        "num_records": len(history),
        "hidden_size": 32,
    }
    meta_path = os.path.join(MODELS_DIR, f"{region}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[train:{region}] saved {model_path}")
    print(f"[train:{region}] saved {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True, help="e.g. MH, KL")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    train_region(args.region.upper(), epochs=args.epochs)
