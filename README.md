# Coin Grading Inference Server

A FastAPI web application for predicting coin grades using ordinal regression.

## Features

- **Dual-image input**: Upload both obverse (front) and reverse (back) images
- **Hough circle preprocessing**: Automatic coin detection and centering
- **Sheldon scale prediction**: Returns grade in standard format (MS64, VF30, etc.)
- **Company conditioning**: Optionally specify grading company (PCGS, NGC, etc.)
- **Mobile-friendly UI**: Camera capture with coin guide overlay

---

## Local Development

### 1. Install dependencies

```bash
cd coin_inference_app
pip install -r requirements.txt
```

### 2. Add your model

```bash
# Copy or symlink your trained model
ln -s ../models/coin_ordinal_best.pth model/coin_ordinal_best.pth
```

### 3. Run the server

```bash
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open in browser

Navigate to http://localhost:8000

---

## Deploy to Heroku

### Prerequisites

1. [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) installed
2. Model file uploaded to cloud storage (S3, Google Drive, Dropbox, etc.)

### Step 1: Upload Model to Cloud Storage

Your model (~230MB) is too large for Heroku's 500MB slug limit. Upload it to:

**Option A: AWS S3**
```bash
aws s3 cp model/coin_ordinal_best.pth s3://your-bucket/coin_ordinal_best.pth --acl public-read
# URL: https://your-bucket.s3.amazonaws.com/coin_ordinal_best.pth
```

**Option B: Google Drive** (make file publicly accessible)
- Upload file, right-click → Get link → Anyone with link
- Use direct download URL: `https://drive.google.com/uc?export=download&id=FILE_ID`

**Option C: Dropbox**
- Upload file, get share link
- Change `?dl=0` to `?dl=1` for direct download

### Step 2: Create Heroku App

```bash
cd coin_inference_app

# Login to Heroku
heroku login

# Create app
heroku create your-coin-grader-app

# Set the model URL
heroku config:set MODEL_URL="https://your-storage.com/coin_ordinal_best.pth"
```

### Step 3: Deploy

```bash
# Initialize git if needed
git init
git add .
git commit -m "Initial commit"

# Deploy to Heroku
git push heroku main
```

### Step 4: Open your app

```bash
heroku open
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MODEL_URL` | URL to download model file | Yes (for Heroku) |
| `PORT` | Server port (set by Heroku) | Auto |

---

## API Endpoints

### `POST /predict`

Predict coin grade from images.

**Form Data:**
- `obverse` (file): Obverse image
- `reverse` (file): Reverse image  
- `company` (optional): Grading company code (PCGS, NGC, etc.)
- `skip_preprocessing` (optional): Skip Hough circle detection

**Response:**
```json
{
  "prediction": "MS64",
  "sheldon_grade": 64,
  "confidence": 85.2,
  "raw_score": 0.8462,
  "company_used": "PCGS",
  "preprocessed": true
}
```

### `POST /preview-preprocessing`

Preview Hough circle preprocessing result.

**Form Data:**
- `image` (file): Coin image

**Response:** JPEG image of preprocessed coin

### `GET /companies`

List available grading companies.

### `GET /grades`

List all valid grades in the model's scale.

### `GET /models`

List available model files.

---

## Project Structure

```
coin_inference_app/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── Procfile            # Heroku process config
├── runtime.txt         # Python version for Heroku
├── .gitignore          # Git ignore rules
├── model/              # Model weights (not in git)
├── model_info/         # Model metadata
└── static/
    └── index.html      # Mobile-friendly web UI
```

---

## Troubleshooting

### "Model not loaded" error
- Set `MODEL_URL` environment variable
- Check model URL is publicly accessible

### Hough circle not detecting coin
- Ensure good lighting and contrast
- Coin should fill 30-70% of the image
- Try the Gallery upload if camera doesn't work

### Heroku H10 (App crashed)
- Check logs: `heroku logs --tail`
- Ensure MODEL_URL is set correctly
- Model download may timeout - try a faster CDN
