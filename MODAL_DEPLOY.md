# Deploy Coin Grader to Modal

Modal is a serverless platform optimized for ML inference. It offers GPU support, pay-per-use pricing, and no model size limits.

## Prerequisites

1. Install Modal CLI:
```bash
pip install modal
```

2. Authenticate (opens browser):
```bash
modal setup
```

## Deploy in 3 Steps

### Step 1: Upload Your Model

```bash
cd /Users/chadstachowicz/coin_scrape

# Create the volume and upload the model
modal volume create coin-grader-models
modal volume put coin-grader-models models/coin_ordinal_best.pth coin_ordinal_best.pth
```

### Step 2: Deploy the App

```bash
cd coin_inference_app
modal deploy modal_app.py
```

### Step 3: Access Your App

After deployment, Modal will show your URL:
```
https://YOUR_USERNAME--numisking-coin-grader-fastapi-app.modal.run
```

## Development Mode

Run locally with hot-reload:
```bash
modal serve modal_app.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/predict` | POST | Grade a coin (multipart form) |
| `/companies` | GET | List grading companies |
| `/grades` | GET | List valid grades |
| `/health` | GET | Health check |

## Example API Call

```bash
curl -X POST "https://YOUR_URL/predict" \
  -F "obverse=@coin_front.jpg" \
  -F "reverse=@coin_back.jpg" \
  -F "company=PCGS"
```

Response:
```json
{
  "prediction": "MS64",
  "sheldon_grade": 64,
  "confidence": 85.2,
  "raw_score": 0.8462,
  "company_used": "PCGS"
}
```

## Configuration

Edit `modal_app.py` to customize:

```python
@app.cls(
    gpu="T4",           # Options: "T4", "A10G", "A100"
    scaledown_window=300,  # Keep warm for 5 min before scaling down
)
@modal.concurrent(max_inputs=10)  # Max concurrent requests per container
```

## Costs

- **GPU (T4)**: ~$0.000164/sec (~$0.59/hr)
- **CPU**: ~$0.000014/sec
- **First 30 seconds/month free**

For a typical coin grading request (~2 seconds):
- ~$0.0003 per request
- ~3,000 requests for $1

## Monitoring

```bash
# View logs
modal app logs numisking-coin-grader

# Check status
modal app list
```

## Troubleshooting

### "Model not found"
```bash
# Check if model is uploaded
modal volume ls coin-grader-models

# Re-upload if needed
modal volume put coin-grader-models models/coin_ordinal_best.pth coin_ordinal_best.pth
```

### Cold starts too slow
Increase `scaledown_window` to keep containers warm longer.

### GPU not available
Check your Modal tier - some GPUs require a paid plan.

