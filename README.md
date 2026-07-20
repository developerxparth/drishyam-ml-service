# Drishyam — Full Project Integration & Deployment Guide

This ties together all four pieces of the project:

```
drishyam/
├── frontend/       (drishyam-frontend — React/Vite, you already have this)
├── backend/        (drishyam-backend — Express/MongoDB, you already have this)
├── ml-service/      NEW — FastAPI + PyTorch LSTM
└── data-pipeline/   NEW — real IMD data pull + import script
```

`ml-service/` and `data-pipeline/` are the two new pieces built here. Drop
your existing `drishyam-frontend` and `drishyam-backend` folders alongside
them (rename the folders to `frontend` and `backend` respectively, or keep
their names — either works, nothing hardcodes the folder name) and you
have one monorepo with everything in it.

## 1. Architecture — how the four pieces actually talk to each other

```
                     ┌─────────────────────┐
                     │   React frontend    │
                     │  (Vite, Tailwind)   │
                     └──────────┬──────────┘
                                │  fetch() -> /api/*
                                ▼
                     ┌─────────────────────┐
                     │  Express backend    │◄──── data-pipeline/
                     │  (Node + Mongoose)  │      (one-time IMD import)
                     └──────────┬──────────┘
                        │                │
                 writes/reads      forwards forecast/
                        │           scenario requests
                        ▼                ▼
                 ┌─────────────┐  ┌─────────────────────┐
                 │  MongoDB    │◄─┤  FastAPI ML service  │
                 │  (Atlas)    │  │  (PyTorch LSTM)      │
                 └─────────────┘  └─────────────────────┘
```

- **Frontend** never talks to Mongo or the ML service directly — only to
  Express, exactly as already built.
- **Express** is the only thing that writes to Mongo. It's also the only
  thing that calls the ML service (`mlService.js` -> `ML_SERVICE_URL`).
- **ML service** only *reads* Mongo (to train), and only ever talks to
  Express, never to the frontend.
- **data-pipeline** is not a running service — it's a script you run once
  (or occasionally) to load real IMD data into the same MongoDB the other
  two pieces use.

Nothing here required changing a single line of your existing frontend or
backend code — every new piece plugs into a seam (`ML_SERVICE_URL`, the
`climaterecords` collection shape) that was already built for exactly this.

## 2. Local integration, step by step

Run these in four terminals, in this order:

```bash
# 0. MongoDB reachable — either local, or have your Atlas cluster ready
#    and MONGODB_URI set consistently in backend/.env, ml-service/.env,
#    and data-pipeline/.env (all three must point at the SAME database)

# 1. Load real data (one-time, or whenever you add more states/years)
cd data-pipeline
pip install -r requirements.txt
cp .env.example .env            # same MONGODB_URI as backend
python fetch_imd_data.py --states MH,KL --start 2005 --end 2024
python import_to_mongo.py --file climate_data.csv

# 2. Train the ML service on the real data you just loaded
cd ../ml-service
pip install -r requirements.txt
cp .env.example .env            # same MONGODB_URI as backend
python -m app.train --region MH
python -m app.train --region KL
uvicorn app.main:app --reload --port 8000
#   -> leave this running

# 3. Backend, pointed at the ML service
cd ../backend
npm install
cp .env.example .env
#   edit .env: set ML_SERVICE_URL=http://localhost:8000
npm run seed                    # fills in synthetic data for any states
                                 # you haven't run the real-data pipeline on yet
npm run dev
#   -> leave this running, http://localhost:4000

# 4. Frontend
cd ../frontend
npm install
cp .env.example .env            # already points at http://localhost:4000/api
npm run dev
#   -> open http://localhost:5173
```

### How to verify it's actually using the real model (not the fallback)

Hit the backend directly and check the `source` field:

```bash
curl http://localhost:4000/api/forecast/MH
```

- `"source": "local-fallback"` → ML service isn't reachable, or that
  region hasn't been trained yet (Express caught the failure and used its
  own trend-estimate math — this is the safety net working as designed,
  not a bug).
- `"source": "ml-service"` → your FastAPI/PyTorch LSTM actually served
  this prediction.

If you expected `ml-service` and got `local-fallback`, check in this
order: (1) is `uvicorn` still running, (2) does `ML_SERVICE_URL` in
`backend/.env` match the port it's on, (3) did you run
`python -m app.train --region MH` for that specific region.

## 3. Pushing to GitHub

```bash
cd drishyam                     # the folder containing all four subfolders
git init
```

Add one root `.gitignore` (each subfolder also has its own, but a root
one catches anything you create at the top level):

```
node_modules/
.env
*.log
venv/
__pycache__/
*.pyc
ml-service/models/*.pt
ml-service/models/*.json
data-pipeline/imd_raw/
data-pipeline/*.csv
```

That last block matters: **do not commit trained model weights or raw
IMD binaries to git** — they're large, regenerable, and not something a
judge needs to clone. Document how to regenerate them instead (this
README already does).

```bash
git add .
git commit -m "Drishyam: MERN + FastAPI/PyTorch LSTM prototype"
```

Create an empty repo on GitHub (github.com/new, no README/gitignore —
you already have them), then:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/drishyam.git
git push -u origin main
```

## 4. Deployment

Straight talk on Vercel here, since you asked specifically: **Vercel is
great for the React frontend, and only the frontend.** It's built for
static sites / serverless functions with short execution limits — not a
great fit for a long-running Express API with a persistent DB connection,
and PyTorch's dependencies are far too large for Vercel's serverless
function size limits. Use Vercel for what it's actually good at, and
Render (also free-tier, made for exactly this) for the other two.

| Piece | Where | Why |
|---|---|---|
| Frontend (React) | **Vercel** | static build, Vercel's actual strength |
| Backend (Express) | **Render** (Web Service) | long-running Node process, easy free tier |
| ML service (FastAPI) | **Render** (Web Service, Docker) or **Hugging Face Spaces** | needs PyTorch installed, Docker is the cleanest path |
| Database | **MongoDB Atlas** | free-tier cluster, already what you're likely using locally |

### 4.1 MongoDB Atlas

If you're not on Atlas yet: create a free cluster at
mongodb.com/cloud/atlas, add a database user, and under Network Access
allow `0.0.0.0/0` (fine for a hackathon demo — tighten it later if this
becomes a real product). Copy the `mongodb+srv://...` connection string —
you'll paste the same one into both Render services below.

### 4.2 Backend → Render

1. render.com → **New +** → **Web Service** → connect your GitHub repo.
2. **Root Directory:** `backend`
3. **Build Command:** `npm install`
4. **Start Command:** `npm start`
5. **Environment variables:**
   - `MONGODB_URI` = your Atlas connection string
   - `CLIENT_ORIGIN` = your Vercel frontend URL (set this after step 4.3,
     then redeploy — or set it to `*` temporarily to unblock yourself)
   - `ML_SERVICE_URL` = your Render ML service URL (set after step 4.4)
6. Deploy. Once it's live, run the seed script once against the deployed
   DB — easiest way is running `npm run seed` locally with `MONGODB_URI`
   in your local `.env` pointed at the *same* Atlas cluster, or add a
   one-off Render "Shell" run of `npm run seed`.
7. Note the resulting URL, e.g. `https://drishyam-backend.onrender.com`.

### 4.3 Frontend → Vercel

1. vercel.com → **Add New** → **Project** → import the same GitHub repo.
2. **Root Directory:** `frontend`
3. Framework preset: Vite (auto-detected)
4. **Environment variable:**
   - `VITE_API_BASE_URL` = `https://drishyam-backend.onrender.com/api`
     (your Render backend URL from step 4.2, + `/api`)
5. Deploy. Note the resulting URL, e.g. `https://drishyam.vercel.app`.
6. Go back to the Render backend's env vars and set `CLIENT_ORIGIN` to
   this exact Vercel URL (needed for CORS), then redeploy the backend.

### 4.4 ML service → Render (Docker)

1. Make sure at least one region is trained **before** deploying — either
   commit the `.pt`/`.json` files for your demo regions temporarily (fine
   for a hackathon, even though the gitignore above excludes them by
   default — you can force-add just your 1-2 demo regions), or add a
   Render "Shell" step after deploy to run
   `python -m app.train --region MH`.
2. render.com → **New +** → **Web Service** → same repo.
3. **Root Directory:** `ml-service`
4. **Runtime:** Docker (Render will detect the `Dockerfile`)
5. **Environment variables:**
   - `MONGODB_URI` = same Atlas string as the backend
6. Deploy. Note the URL, e.g. `https://drishyam-ml.onrender.com`.
7. Go back to the Render **backend** service's env vars, set
   `ML_SERVICE_URL` to this URL, redeploy the backend.

Render's free tier spins down services after inactivity — the first
request after idling can take 30-60s to wake up. For a live judging demo,
open all three Render/Vercel URLs a few minutes beforehand to warm them
up, and keep the local-fallback safety net in mind as your plan B if a
service is slow to respond mid-demo.

### 4.5 Final checklist before submission

- [ ] Open the deployed Vercel URL fresh (private/incognito window) and
      confirm it shows the `LIVE` badge, not `OFFLINE (mock)`.
- [ ] Check `GET https://<backend>.onrender.com/api/forecast/MH` directly
      and confirm `"source": "ml-service"`.
- [ ] Click through the what-if sliders on the deployed site and confirm
      the numbers actually change.
- [ ] Note 2-3 known-good slider positions so you're not improvising
      values live in front of judges.
- [ ] Make sure the GitHub repo, deployed link, and PPT all describe the
      same architecture consistently.

## 5. What's still genuinely mocked/simplified, and why (worth disclosing)

Being upfront about scope in your submission reads as strength, not
weakness, to ISRO/IMD judges:

- Real IMD data is loaded only for the states you ran `fetch_imd_data.py`
  for; other states still use the synthetic seed generator.
- The LSTM is univariate (rainfall only) and one-step-ahead, trained per
  region — documented in `ml-service/README.md`.
- Explainability (SHAP) isn't wired into the ML service in this pass —
  the original plan's approach (precompute offline, store in the
  `Explanation` collection, serve as a cached read) is still the right
  call for demo-day stability and is unchanged from before.
