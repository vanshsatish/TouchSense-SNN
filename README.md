# TouchSense-SNN backend

FastAPI research backend for touchscreen gesture feature extraction and SNN experiments.

## What this backend does
- Accepts pointer/touch trajectories from a frontend.
- Extracts tap interval, mean speed, acceleration magnitude, path deviation, pressure summary, and quality/context fields.
- Runs a small Leaky Integrate-and-Fire SNN if a trained checkpoint exists.
- Returns features without inventing a motor-function diagnosis.
- Includes a starter training script for properly labeled research data.

## Important limitations
This is research software, not a medical device. The initial model is untrained; `/analyze` returns `prediction: null` until a checkpoint is trained and placed at `artifacts/touch_snn.pt`. Placeholder classes are not clinical categories. Pointer pressure is browser/device-dependent and may be unavailable. Acceleration is estimated from sampled position and timestamps, not measured by a physical accelerometer.

## Run locally
```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## Example request
```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "tap_interval_ms": 280,
    "screen_width": 390,
    "screen_height": 844,
    "points": [
      {"x": 20, "y": 30, "t_ms": 0, "pressure": 0.4},
      {"x": 35, "y": 38, "t_ms": 16, "pressure": 0.5},
      {"x": 55, "y": 50, "t_ms": 32, "pressure": 0.5}
    ]
  }'
```

## Train a model
Prepare a CSV with the five feature columns and integer `label` values 0, 1, or 2. See `data/sample_schema.csv`.

```bash
python scripts/train.py --csv path/to/consented_labeled_data.csv --epochs 30
```

The included split is only a smoke test. For research claims, split by participant (and preferably by device/site for external validation), pre-register outcomes, report confidence intervals and class-wise metrics, and compare with non-SNN baselines. Never use random gesture-level splitting as the primary evidence of generalization.

## Privacy and deployment
The API does not persist submitted gesture data. Add authentication, TLS, rate limiting, access controls, retention rules, and restrictive CORS before deployment. Do not commit identifiable or raw participant data or model checkpoints to a public repository. Obtain appropriate consent and institutional review where applicable.
