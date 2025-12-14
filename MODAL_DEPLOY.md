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

## Deploy in 4 Steps

### Step 1: Upload Your Models

```bash
cd /Users/chadstachowicz/coin_scrape

# Create the volume and upload the models
modal volume create coin-grader-models

# Upload ResNet model (standard)
modal volume put coin-grader-models models/coin_ordinal_best.pth coin_ordinal_best.pth

# Upload ConvNeXt model (advanced) - optional but recommended
modal volume put coin-grader-models models/coin_convnext_best.pth coin_convnext_best.pth
```

### Step 2: Set Up API Keys (for programmatic access)

```bash
# Create API key for JSON API access
modal secret create api_keys API_KEY="your-secure-random-api-key"

# For multiple API keys (e.g., different clients):
modal secret create api_keys API_KEY="key1,key2,key3"
```

### Step 3: Deploy the App

```bash
cd coin_inference_app
modal deploy modal_app.py
```

### Step 4: Access Your App

After deployment, Modal will show your URL:
```
https://YOUR_USERNAME--numisking-coin-grader-fastapi-app.modal.run
```

## Development Mode

Run locally with hot-reload:
```bash
modal serve modal_app.py
```

## Available Models

| Model ID | Backbone | Description |
|----------|----------|-------------|
| `standard` | ResNet-50 | Fast and reliable (default) |
| `advanced` | ConvNeXt-Small | Higher accuracy |

## API Endpoints

### Web Interface Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | None | Web interface |
| `/predict` | POST | None | Grade a coin (multipart form) |
| `/split-combined` | POST | None | Split TrueView image |
| `/models` | GET | None | List available models |
| `/companies` | GET | None | List grading companies |
| `/grades` | GET | None | List valid grades |
| `/health` | GET | None | Health check |

### JSON API Endpoints (API Key Required)

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/predict` | POST | API Key | Grade a coin (JSON) |
| `/api/models` | GET | API Key | List available models |
| `/api/companies` | GET | API Key | List grading companies |
| `/api/health` | GET | API Key | Health check with auth |

## Example: Web Form Upload

```bash
curl -X POST "https://YOUR_URL/predict" \
  -F "obverse=@coin_front.jpg" \
  -F "reverse=@coin_back.jpg" \
  -F "model=standard" \
  -F "company=PCGS"
```

Response:
```json
{
  "prediction": "MS64",
  "sheldon_grade": 64,
  "confidence": 85.2,
  "raw_score": 0.8462,
  "company_used": "PCGS",
  "model_used": "Standard"
}
```

## Example: JSON API (Programmatic Access)

### cURL

```bash
# First, base64 encode your images
OBVERSE_B64=$(base64 -i coin_front.jpg)
REVERSE_B64=$(base64 -i coin_back.jpg)

curl -X POST "https://YOUR_URL/api/predict" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d "{
    \"obverse_base64\": \"$OBVERSE_B64\",
    \"reverse_base64\": \"$REVERSE_B64\",
    \"model\": \"advanced\",
    \"company\": \"PCGS\"
  }"
```

### Python

```python
import requests
import base64

API_URL = "https://YOUR_URL/api/predict"
API_KEY = "your-api-key"

# Load and encode images
with open("coin_front.jpg", "rb") as f:
    obverse_b64 = base64.b64encode(f.read()).decode()
with open("coin_back.jpg", "rb") as f:
    reverse_b64 = base64.b64encode(f.read()).decode()

# Make prediction request
response = requests.post(
    API_URL,
    headers={"X-API-Key": API_KEY},
    json={
        "obverse_base64": obverse_b64,
        "reverse_base64": reverse_b64,
        "model": "advanced",  # or "standard"
        "company": "PCGS"
    }
)

result = response.json()
print(f"Grade: {result['prediction']} (Sheldon: {result['sheldon_grade']})")
print(f"Confidence: {result['confidence']}%")
print(f"Model: {result['model_used']}")
```

### JavaScript/Node.js

```javascript
const fs = require('fs');

const API_URL = 'https://YOUR_URL/api/predict';
const API_KEY = 'your-api-key';

// Load and encode images
const obverseB64 = fs.readFileSync('coin_front.jpg').toString('base64');
const reverseB64 = fs.readFileSync('coin_back.jpg').toString('base64');

fetch(API_URL, {
  method: 'POST',
  headers: {
    'X-API-Key': API_KEY,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    obverse_base64: obverseB64,
    reverse_base64: reverseB64,
    model: 'standard',
    company: 'PCGS'
  })
})
.then(res => res.json())
.then(result => {
  console.log(`Grade: ${result.prediction}`);
  console.log(`Confidence: ${result.confidence}%`);
});
```

### Response Format

```json
{
  "prediction": "MS64",
  "sheldon_grade": 64,
  "confidence": 87.5,
  "raw_score": 0.8234,
  "company_used": "PCGS",
  "model_used": "Advanced"
}
```

### Error Responses

```json
// 401 Unauthorized - Missing or invalid API key
{
  "error": "Invalid API key"
}

// 400 Bad Request - Invalid images
{
  "error": "Invalid base64 encoding for obverse image: ..."
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

