"""
Coin Grading Inference Server

FastAPI app for coin grade prediction using ordinal regression.
Accepts obverse and reverse images, returns predicted Sheldon grade.

Preprocessing pipeline:
1. Hough circle detection to find the coin
2. Crop to the detected circle
3. Center on white background
4. Resize for model input
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from torchvision import transforms
import json
import os
import io
from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional
import numpy as np
import cv2
import urllib.request
import tempfile

# ============================================================================
# MODEL ARCHITECTURE (must match training)
# ============================================================================

class OrdinalRegressionResNet(nn.Module):
    """ResNet-50 based model for ordinal regression on coin grades.
    
    MUST match the architecture used during training exactly!
    """
    
    def __init__(self, num_companies=None, company_embedding_dim=32, freeze_backbone=False):
        super(OrdinalRegressionResNet, self).__init__()
        
        self.use_company = num_companies is not None
        
        # Company embedding
        if self.use_company:
            self.company_embedding = nn.Embedding(num_companies, company_embedding_dim)
        
        # ResNet-50 backbone for obverse
        import torchvision.models as models
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.obverse_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # ResNet-50 backbone for reverse (separate weights)
        resnet2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.reverse_encoder = nn.Sequential(*list(resnet2.children())[:-1])
        
        if freeze_backbone:
            for param in self.obverse_encoder.parameters():
                param.requires_grad = False
            for param in self.reverse_encoder.parameters():
                param.requires_grad = False
        
        self.feature_dim = 2048
        
        # Fusion layer - MUST match training architecture
        fusion_input_dim = self.feature_dim * 2  # Two ResNet outputs
        if self.use_company:
            fusion_input_dim += company_embedding_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Regression head - MUST match training architecture
        self.regression_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, obverse, reverse, company_idx=None):
        obverse_feat = self.obverse_encoder(obverse).view(obverse.size(0), -1)
        reverse_feat = self.reverse_encoder(reverse).view(reverse.size(0), -1)
        
        combined = torch.cat([obverse_feat, reverse_feat], dim=1)
        
        if self.use_company and company_idx is not None:
            company_emb = self.company_embedding(company_idx)
            combined = torch.cat([combined, company_emb], dim=1)
        
        fused = self.fusion(combined)
        output = self.regression_head(fused).squeeze(-1)
        
        return output


# ============================================================================
# COIN PREPROCESSING - Hough Circle Detection
# ============================================================================

def preprocess_coin_image(pil_image, output_size=512, debug=False):
    """
    Preprocess coin image:
    1. Detect coin using Hough Circle Transform
    2. Crop to the detected circle with padding
    3. Center on white background
    
    Args:
        pil_image: PIL Image
        output_size: Output image size (square)
        debug: If True, return debug info
    
    Returns:
        Preprocessed PIL Image (or tuple with debug info)
    """
    # Convert PIL to OpenCV format
    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # Get image dimensions
    height, width = img_bgr.shape[:2]
    min_dim = min(height, width)
    
    # Convert to grayscale for circle detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Hough Circle Transform parameters
    # - dp: Inverse ratio of accumulator resolution (1 = same as input)
    # - minDist: Minimum distance between circle centers
    # - param1: Upper threshold for Canny edge detector
    # - param2: Accumulator threshold (lower = more circles detected)
    # - minRadius/maxRadius: Expected coin size range
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dim // 2,  # Only find one main circle
        param1=50,
        param2=30,
        minRadius=int(min_dim * 0.2),  # Coin should be at least 20% of image
        maxRadius=int(min_dim * 0.5)   # And at most 50% (with some margin)
    )
    
    # If no circle found, try with more lenient parameters
    if circles is None:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dim // 3,
            param1=100,
            param2=20,  # More lenient
            minRadius=int(min_dim * 0.15),
            maxRadius=int(min_dim * 0.55)
        )
    
    # Default to center crop if no circle detected
    if circles is None:
        print("  ⚠ No circle detected, using center crop")
        cx, cy = width // 2, height // 2
        radius = min_dim // 2 - 10
    else:
        # Get the best circle (first one, usually the strongest)
        circles = np.uint16(np.around(circles))
        cx, cy, radius = circles[0][0]
        print(f"  ✓ Circle detected: center=({cx}, {cy}), radius={radius}")
    
    # Add padding around the coin (5% of radius)
    padding = int(radius * 0.05)
    crop_radius = radius + padding
    
    # Calculate crop boundaries
    x1 = max(0, int(cx - crop_radius))
    y1 = max(0, int(cy - crop_radius))
    x2 = min(width, int(cx + crop_radius))
    y2 = min(height, int(cy + crop_radius))
    
    # Crop the coin region
    cropped = img_bgr[y1:y2, x1:x2]
    
    # Create white background
    white_bg = np.ones((output_size, output_size, 3), dtype=np.uint8) * 255
    
    # Resize cropped coin to fit in output
    crop_h, crop_w = cropped.shape[:2]
    scale = (output_size * 0.95) / max(crop_h, crop_w)  # 95% of output size
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Center on white background
    x_offset = (output_size - new_w) // 2
    y_offset = (output_size - new_h) // 2
    white_bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Optional: Create circular mask to clean up edges
    mask = np.zeros((output_size, output_size), dtype=np.uint8)
    center = output_size // 2
    cv2.circle(mask, (center, center), int(output_size * 0.47), 255, -1)
    
    # Apply slight feathering to mask edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    
    # Blend with white background using mask
    white_full = np.ones_like(white_bg, dtype=float) * 255
    result = (white_bg.astype(float) * mask_3ch + white_full * (1 - mask_3ch)).astype(np.uint8)
    
    # Convert back to PIL (RGB)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    pil_result = Image.fromarray(result_rgb)
    
    if debug:
        return pil_result, {'cx': cx, 'cy': cy, 'radius': radius, 'detected': circles is not None}
    
    return pil_result


# ============================================================================
# GLOBALS & CONFIG
# ============================================================================

# Default valid grades (will be loaded from checkpoint)
VALID_GRADES = [2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45,
                50, 53, 55, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68]
NUM_STEPS = len(VALID_GRADES)
STEP_TO_GRADE = {step: grade for step, grade in enumerate(VALID_GRADES)}
ENCODING_TYPE = 'step_based'
GRADE_MIN = 1.0
GRADE_MAX = 70.0

# Model state
model = None
company_to_idx = None
idx_to_company = None
use_company = False
IMAGE_SIZE = 512

# Image transforms (must match training)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def step_to_sheldon(step):
    """Convert step position back to Sheldon grade."""
    step = int(round(step))
    step = max(0, min(step, NUM_STEPS - 1))
    return STEP_TO_GRADE[step]


def denormalize_grade(normalized_grade):
    """Convert normalized grade back to Sheldon scale."""
    if ENCODING_TYPE == 'step_based':
        step = normalized_grade * (NUM_STEPS - 1)
        return step_to_sheldon(step)
    else:
        return normalized_grade * (GRADE_MAX - GRADE_MIN) + GRADE_MIN


def format_grade_name(sheldon_grade):
    """Convert Sheldon number to grade name (MS64, VF30, etc)."""
    grade = int(round(sheldon_grade))
    if grade >= 60:
        return f"MS{grade}"
    elif grade >= 50:
        return f"AU{grade}"
    elif grade >= 40:
        return f"XF{grade}"
    elif grade >= 20:
        return f"VF{grade}"
    elif grade >= 8:
        return f"VG{grade:02d}"
    elif grade >= 4:
        return f"G{grade:02d}"
    elif grade == 3:
        return "AG03"
    elif grade == 2:
        return "FR02"
    else:
        return f"P{grade:02d}"


def load_model(model_path: str):
    """Load trained ordinal regression model."""
    global model, company_to_idx, idx_to_company, use_company
    global VALID_GRADES, NUM_STEPS, STEP_TO_GRADE, ENCODING_TYPE, GRADE_MIN, GRADE_MAX
    
    print(f"Loading model from: {model_path}")
    
    device = torch.device('cpu')  # Use CPU for inference server
    checkpoint = torch.load(model_path, map_location=device)
    
    use_company = checkpoint.get('use_company', False)
    company_to_idx = checkpoint.get('company_to_idx')
    idx_to_company = checkpoint.get('idx_to_company')
    company_embedding_dim = checkpoint.get('company_embedding_dim', 32)
    
    # Detect encoding type
    ENCODING_TYPE = checkpoint.get('encoding', 'sheldon')
    
    if ENCODING_TYPE == 'step_based':
        VALID_GRADES = checkpoint.get('valid_grades', VALID_GRADES)
        NUM_STEPS = checkpoint.get('num_steps', len(VALID_GRADES))
        STEP_TO_GRADE = {step: grade for step, grade in enumerate(VALID_GRADES)}
        print(f"  Encoding: STEP-BASED ({NUM_STEPS} steps)")
    else:
        GRADE_MIN = checkpoint.get('grade_min', 1.0)
        GRADE_MAX = checkpoint.get('grade_max', 70.0)
        print(f"  Encoding: SHELDON-BASED ({GRADE_MIN}-{GRADE_MAX})")
    
    num_companies = len(company_to_idx) if use_company and company_to_idx else None
    
    model = OrdinalRegressionResNet(
        num_companies=num_companies,
        company_embedding_dim=company_embedding_dim,
        freeze_backbone=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded (Epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"  Best val MAE: {checkpoint.get('val_mae', 'unknown')}")
    print(f"  Company conditioning: {use_company}")
    if use_company and company_to_idx:
        print(f"  Companies: {list(company_to_idx.keys())}")
    
    return model


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Coin Grading API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_DIRECTORY = "model"
MODEL_INFO_DIRECTORY = "model_info"


def download_model_from_url(url: str, dest_path: str):
    """Download model from URL (e.g., S3, Google Drive, Dropbox)."""
    print(f"Downloading model from: {url}")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"✓ Model downloaded to: {dest_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup. Supports local file or remote URL."""
    
    # Check for model URL in environment (for Heroku/cloud deployment)
    model_url = os.environ.get("MODEL_URL")
    model_path = os.path.join(MODEL_DIRECTORY, "coin_ordinal_best.pth")
    
    # Ensure model directory exists
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)
    
    if model_url and not os.path.exists(model_path):
        # Download model from URL
        download_model_from_url(model_url, model_path)
    
    if os.path.exists(model_path):
        load_model(model_path)
    else:
        print(f"⚠️ Model not found: {model_path}")
        print("  Set MODEL_URL environment variable or upload a model")


@app.get("/")
def read_root():
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict(
    obverse: UploadFile = File(..., description="Obverse (front) image of coin"),
    reverse: UploadFile = File(..., description="Reverse (back) image of coin"),
    company: Optional[str] = Form(None, description="Grading company (PCGS, NGC, etc)"),
    skip_preprocessing: bool = Form(False, description="Skip Hough circle preprocessing")
):
    """
    Predict coin grade from obverse and reverse images.
    
    Preprocessing pipeline:
    1. Hough circle detection to find the coin
    2. Crop and center on white background
    3. Run through model
    
    Returns predicted Sheldon grade and grade name (e.g., MS64).
    """
    global model, company_to_idx, use_company
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Upload a model first.")
    
    try:
        # Read images
        obverse_bytes = await obverse.read()
        reverse_bytes = await reverse.read()
        
        obverse_img = Image.open(io.BytesIO(obverse_bytes)).convert("RGB")
        reverse_img = Image.open(io.BytesIO(reverse_bytes)).convert("RGB")
        
        # Preprocess: Hough circle detection + white background
        if not skip_preprocessing:
            print("Preprocessing obverse...")
            obverse_img = preprocess_coin_image(obverse_img, output_size=IMAGE_SIZE)
            print("Preprocessing reverse...")
            reverse_img = preprocess_coin_image(reverse_img, output_size=IMAGE_SIZE)
        
        # Transform for model
        obverse_tensor = transform(obverse_img).unsqueeze(0)
        reverse_tensor = transform(reverse_img).unsqueeze(0)
        
        # Handle company conditioning
        company_idx_tensor = None
        company_name_used = None
        
        if use_company and company_to_idx:
            if company and company.upper() in company_to_idx:
                # User provided a valid company
                company_idx_tensor = torch.tensor([company_to_idx[company.upper()]])
                company_name_used = company.upper()
            else:
                # Model requires company but none provided - default to PCGS or first available
                default_company = 'PCGS' if 'PCGS' in company_to_idx else list(company_to_idx.keys())[0]
                company_idx_tensor = torch.tensor([company_to_idx[default_company]])
                company_name_used = f"{default_company} (default)"
                print(f"  Using default company: {default_company}")
        
        # Predict
        with torch.no_grad():
            prediction = model(obverse_tensor, reverse_tensor, company_idx_tensor)
            normalized_pred = prediction.item()
            
            # Denormalize to Sheldon grade
            sheldon_grade = denormalize_grade(normalized_pred)
            grade_name = format_grade_name(sheldon_grade)
        
        # Get confidence (based on how close to a step boundary)
        if ENCODING_TYPE == 'step_based':
            step_pos = normalized_pred * (NUM_STEPS - 1)
            confidence = 1.0 - abs(step_pos - round(step_pos))  # Higher when closer to step
            confidence = max(0.5, confidence)  # Floor at 50%
        else:
            confidence = 0.85  # Default confidence for non-step models
        
        return {
            "prediction": grade_name,
            "sheldon_grade": int(sheldon_grade),
            "confidence": round(confidence * 100, 1),
            "raw_score": round(normalized_pred, 4),
            "company_used": company_name_used,
            "preprocessed": not skip_preprocessing
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing images: {str(e)}")


@app.get("/models")
async def list_models():
    """List available model files."""
    try:
        files = os.listdir(MODEL_DIRECTORY)
        model_files = [f for f in files if f.endswith('.pth')]
        return JSONResponse(content=model_files)
    except FileNotFoundError:
        return JSONResponse(content={"error": "Models directory not found."}, status_code=404)


@app.post("/load-model")
async def load_model_endpoint(model_name: str = Query(..., description="Model filename")):
    """Load a specific model."""
    model_path = os.path.join(MODEL_DIRECTORY, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    try:
        load_model(model_path)
        return {"status": "success", "message": f"Loaded model: {model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.get("/companies")
async def list_companies():
    """List available grading companies (if model supports company conditioning)."""
    if use_company and company_to_idx:
        return {"companies": list(company_to_idx.keys())}
    return {"companies": [], "message": "Model does not use company conditioning"}


@app.get("/grades")
async def list_grades():
    """List valid grades in the model's grading scale."""
    grade_info = []
    for grade in VALID_GRADES:
        grade_info.append({
            "sheldon": grade,
            "name": format_grade_name(grade)
        })
    return {"grades": grade_info, "encoding": ENCODING_TYPE}


@app.post("/preview-preprocessing")
async def preview_preprocessing(
    image: UploadFile = File(..., description="Coin image to preprocess")
):
    """
    Preview the Hough circle preprocessing result.
    
    Returns the preprocessed image as JPEG.
    Useful for debugging preprocessing issues.
    """
    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess with debug info
        processed, debug_info = preprocess_coin_image(img, output_size=IMAGE_SIZE, debug=True)
        
        # Convert to bytes
        output_buffer = io.BytesIO()
        processed.save(output_buffer, format='JPEG', quality=90)
        output_buffer.seek(0)
        
        return StreamingResponse(
            output_buffer,
            media_type="image/jpeg",
            headers={
                "X-Circle-Detected": str(debug_info.get('detected', False)),
                "X-Circle-Center-X": str(debug_info.get('cx', 0)),
                "X-Circle-Center-Y": str(debug_info.get('cy', 0)),
                "X-Circle-Radius": str(debug_info.get('radius', 0))
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing image: {str(e)}")


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

