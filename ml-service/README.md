# Drishyam — ML Service (FastAPI + PyTorch LSTM)

This is the real AI engine from your PPT's tech-stack slide: a small,
per-region LSTM trained on the real IMD rainfall history already sitting
in your MongoDB (loaded via `data-pipeline/`). It plugs directly into
the seam your Express backend already has — `ML_SERVICE_URL` — with
**zero backend code changes**.

## How it fits with the rest of the stack

```
[React]  ->  [Express + MongoDB]  --(REST)-->  [FastAPI + PyTorch LSTM]
                    ^                                    |
                    |____________ reads climaterecords ___|
                                  directly, read-only
```

- Express still owns all writes (Forecast/Scenario/RiskReport documents).
- This service only *reads* `climaterecords` (at training time) and never
  writes to Mongo — it's a pure prediction engine Express calls over HTTP.
- If this service is down, unset, or times out, Express's existing
  `mlService.js` fallback logic kicks in automatically — you built that
  already, this doesn't change it.

## Setup

```bash
cd ml-service
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # same MONGODB_URI as drishyam-backend
```

## Train at least one region

You need real data in MongoDB first (see `../data-pipeline`). Then:

```bash
python -m app.train --region MH
python -m app.train --region KL
```

This reads `climaterecords` for that region, trains a small LSTM
(univariate, 12-month input window -> next month prediction, ~300
epochs, trains in seconds/minutes on CPU), and saves two files per
region under `models/`:

- `{region}_lstm.pt` — the trained weights
- `{region}_meta.json` — normalization range, per-month historical
  means, and the most recent 12-month window used at inference time

## Run the service

```bash
uvicorn app.main:app --reload --port 8000
```

Check it:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"region":"MH","month":"Jul"}'
```

## Wire it to the Express backend

In `drishyam-backend/.env`:

```
ML_SERVICE_URL=http://localhost:8000
```

Restart the backend. Hit any of its endpoints that trigger a forecast
(`/api/forecast/MH`, `/api/scenario`) and check the `source` field in the
response — it should now say `"ml-service"` instead of `"local-fallback"`.
That flip is your proof the real LSTM is in the loop.

## Endpoints

| Method | Route | Body | Notes |
|---|---|---|---|
| GET | `/health` | — | lists which regions have trained models |
| GET | `/regions` | — | same list, just the array |
| POST | `/predict` | `{ region, month }` | 404 if region isn't trained yet (Express falls back automatically) |
| POST | `/scenario` | `{ region, rainfallDelta, tempDelta }` | same 404-fallback behavior |

## Design decisions worth knowing (and worth mentioning to judges)

- **Univariate LSTM.** The model only looks at rainfall history, not
  temperature, to stay small and trainable in minutes on a laptop — matching
  the original scope. `tempDelta` in `/scenario` is applied as a
  physically-motivated multiplier (more heat -> more evapotranspiration ->
  less effective rainfall) rather than being a model input.
- **One-step-ahead forecasting.** The model predicts the month right after
  its most recent 12 months on record — it doesn't take an arbitrary future
  month as input. This matches how `mlService.js` actually calls it (always
  "the current month"), but is worth being upfront about if asked how far
  ahead the model truly forecasts.
- **Per-region models**, not one global model — simpler to train, explain,
  and debug for a hackathon timeline, at the cost of not sharing patterns
  across states. A natural "future work" line for your submission.
- **Explainability (SHAP)** isn't wired up here — see the top-level
  `README.md` for the recommended precompute-offline approach, since live
  SHAP against an LSTM adds real latency you don't want live during judging.

## Deploying

A `Dockerfile` is included. See the top-level `README.md` for exact
Render/Hugging Face Spaces steps and required environment variables.
