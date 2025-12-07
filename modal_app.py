"""
Coin Grading Inference - Modal Deployment

Deploy with:
    modal deploy modal_app.py

Run locally:
    modal serve modal_app.py

Setup database secret (one-time):
    modal secret create neon-db DATABASE_URL="postgresql://..."

Test endpoint:
    curl -X POST "https://YOUR_APP_URL/predict" \
        -F "obverse=@coin_front.jpg" \
        -F "reverse=@coin_back.jpg"
"""

import modal
import io
import time
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================================
# MODAL APP CONFIGURATION
# ============================================================================

# Create the Modal app
app = modal.App("numisking-coin-grader")

# Create a volume to store the model
model_volume = modal.Volume.from_name("coin-grader-models", create_if_missing=True)
MODEL_DIR = "/models"

# Create a volume to store prediction images
images_volume = modal.Volume.from_name("coin-grader-images", create_if_missing=True)
IMAGES_DIR = "/images"

# Database secret (create with: modal secret create neon-db DATABASE_URL="postgresql://...")
db_secret = modal.Secret.from_name("neon-db")

# Admin credentials secret
admin_secret = modal.Secret.from_name("custom-secret")

# Get path to static files
LOCAL_STATIC_DIR = Path(__file__).parent / "static"

# Define the container image with static files
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",  # Pin numpy 1.x for torch compatibility
        "torch==2.2.0",
        "torchvision==0.17.0",
        "Pillow>=10.0.0",
        "opencv-python-headless>=4.8.0",
        "fastapi>=0.100.0",
        "python-multipart>=0.0.6",
        "psycopg2-binary>=2.9.0",  # PostgreSQL driver
    )
    .add_local_dir(LOCAL_STATIC_DIR, remote_path="/app/static")
)


# ============================================================================
# DATABASE HELPERS
# ============================================================================

def get_db_connection():
    """Get a database connection from the pool."""
    import os
    import psycopg2
    
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return None
    
    try:
        conn = psycopg2.connect(database_url)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None


def init_database():
    """Initialize database tables if they don't exist."""
    conn = get_db_connection()
    if not conn:
        print("⚠️ No database connection - analytics disabled")
        return False
    
    try:
        cur = conn.cursor()
        
        # Predictions table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT NOW(),
                ip_address VARCHAR(45),
                user_agent TEXT,
                
                predicted_grade VARCHAR(10),
                sheldon_grade INTEGER,
                confidence FLOAT,
                company_used VARCHAR(20),
                raw_score FLOAT,
                processing_time_ms INTEGER,
                
                async_mode BOOLEAN DEFAULT FALSE,
                job_id VARCHAR(100),
                has_images BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Add has_images column if it doesn't exist (for existing tables)
        cur.execute("""
            DO $$ 
            BEGIN 
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                               WHERE table_name='predictions' AND column_name='has_images') 
                THEN 
                    ALTER TABLE predictions ADD COLUMN has_images BOOLEAN DEFAULT FALSE;
                END IF;
            END $$;
        """)
        
        # Split requests table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS split_requests (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT NOW(),
                ip_address VARCHAR(45),
                user_agent TEXT,
                
                detection_method VARCHAR(20),
                coins_found INTEGER,
                processing_time_ms INTEGER,
                has_images BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Add has_images column to split_requests if it doesn't exist
        cur.execute("""
            DO $$ 
            BEGIN 
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                               WHERE table_name='split_requests' AND column_name='has_images') 
                THEN 
                    ALTER TABLE split_requests ADD COLUMN has_images BOOLEAN DEFAULT FALSE;
                END IF;
            END $$;
        """)
        
        # Indexes for analytics queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_created_at 
            ON predictions(created_at)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_ip 
            ON predictions(ip_address)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_split_created_at 
            ON split_requests(created_at)
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Database tables initialized")
        return True
    except Exception as e:
        print(f"Database init error: {e}")
        conn.close()
        return False


def log_prediction(ip: str, user_agent: str, result: dict, processing_time_ms: int, 
                   async_mode: bool = False, job_id: str = None, has_images: bool = False):
    """Log a prediction to the database. Returns the prediction ID."""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO predictions 
            (ip_address, user_agent, predicted_grade, sheldon_grade, confidence, 
             company_used, raw_score, processing_time_ms, async_mode, job_id, has_images)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            ip,
            user_agent[:500] if user_agent else None,  # Truncate long user agents
            result.get("prediction"),
            result.get("sheldon_grade"),
            result.get("confidence"),
            result.get("company_used"),
            result.get("raw_score"),
            processing_time_ms,
            async_mode,
            job_id,
            has_images
        ))
        prediction_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return prediction_id
    except Exception as e:
        print(f"Error logging prediction: {e}")
        conn.close()
        return None


def save_prediction_images(prediction_id: int, obverse_bytes: bytes, reverse_bytes: bytes):
    """Save prediction images to the images volume."""
    import os
    
    try:
        # Create directory for this prediction
        pred_dir = f"{IMAGES_DIR}/{prediction_id}"
        os.makedirs(pred_dir, exist_ok=True)
        
        # Save images
        with open(f"{pred_dir}/obverse.jpg", "wb") as f:
            f.write(obverse_bytes)
        
        with open(f"{pred_dir}/reverse.jpg", "wb") as f:
            f.write(reverse_bytes)
        
        return True
    except Exception as e:
        print(f"Error saving images: {e}")
        return False


def get_prediction_image(prediction_id: int, side: str) -> bytes:
    """Get a prediction image from the images volume."""
    import os
    
    try:
        image_path = f"{IMAGES_DIR}/{prediction_id}/{side}.jpg"
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                return f.read()
        return None
    except Exception as e:
        print(f"Error reading image: {e}")
        return None


def log_split_request(ip: str, user_agent: str, result: dict, processing_time_ms: int, 
                      has_images: bool = False):
    """Log a split request to the database. Returns the split request ID."""
    conn = get_db_connection()
    if not conn:
        return None

    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO split_requests
            (ip_address, user_agent, detection_method, coins_found, processing_time_ms, has_images)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            ip,
            user_agent[:500] if user_agent else None,
            result.get("detection_method"),
            result.get("coins_found"),
            processing_time_ms,
            has_images
        ))
        split_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return split_id
    except Exception as e:
        print(f"Error logging split request: {e}")
        conn.close()
        return None


def save_split_images(split_id: int, original_bytes: bytes, obverse_bytes: bytes, reverse_bytes: bytes):
    """Save split request images to the images volume."""
    import os
    
    try:
        # Create directory for this split request
        split_dir = f"{IMAGES_DIR}/splits/{split_id}"
        os.makedirs(split_dir, exist_ok=True)
        
        # Save original TrueView image
        with open(f"{split_dir}/original.jpg", "wb") as f:
            f.write(original_bytes)
        
        # Save split images
        with open(f"{split_dir}/obverse.jpg", "wb") as f:
            f.write(obverse_bytes)
        
        with open(f"{split_dir}/reverse.jpg", "wb") as f:
            f.write(reverse_bytes)
        
        return True
    except Exception as e:
        print(f"Error saving split images: {e}")
        return False


def get_analytics_stats():
    """Get analytics statistics."""
    conn = get_db_connection()
    if not conn:
        return {"error": "Database not connected"}
    
    try:
        cur = conn.cursor()
        stats = {}
        
        # Total predictions
        cur.execute("SELECT COUNT(*) FROM predictions")
        stats["total_predictions"] = cur.fetchone()[0]
        
        # Total splits
        cur.execute("SELECT COUNT(*) FROM split_requests")
        stats["total_splits"] = cur.fetchone()[0]
        
        # Predictions today
        cur.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE created_at >= CURRENT_DATE
        """)
        stats["predictions_today"] = cur.fetchone()[0]
        
        # Predictions last 7 days
        cur.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
        """)
        stats["predictions_7d"] = cur.fetchone()[0]
        
        # Unique IPs today
        cur.execute("""
            SELECT COUNT(DISTINCT ip_address) FROM predictions 
            WHERE created_at >= CURRENT_DATE
        """)
        stats["unique_ips_today"] = cur.fetchone()[0]
        
        # Unique IPs all time
        cur.execute("SELECT COUNT(DISTINCT ip_address) FROM predictions")
        stats["unique_ips_total"] = cur.fetchone()[0]
        
        # Average confidence
        cur.execute("SELECT AVG(confidence) FROM predictions WHERE confidence IS NOT NULL")
        avg_conf = cur.fetchone()[0]
        stats["avg_confidence"] = round(avg_conf, 1) if avg_conf else None
        
        # Grade distribution (top 10)
        cur.execute("""
            SELECT predicted_grade, COUNT(*) as cnt 
            FROM predictions 
            WHERE predicted_grade IS NOT NULL
            GROUP BY predicted_grade 
            ORDER BY cnt DESC 
            LIMIT 10
        """)
        stats["grade_distribution"] = [{"grade": r[0], "count": r[1]} for r in cur.fetchall()]
        
        # Top IPs (top 10)
        cur.execute("""
            SELECT ip_address, COUNT(*) as cnt 
            FROM predictions 
            WHERE ip_address IS NOT NULL
            GROUP BY ip_address 
            ORDER BY cnt DESC 
            LIMIT 10
        """)
        stats["top_ips"] = [{"ip": r[0], "count": r[1]} for r in cur.fetchall()]
        
        # Average processing time
        cur.execute("SELECT AVG(processing_time_ms) FROM predictions WHERE processing_time_ms IS NOT NULL")
        avg_time = cur.fetchone()[0]
        stats["avg_processing_time_ms"] = round(avg_time, 1) if avg_time else None
        
        # Predictions per hour (last 24 hours)
        cur.execute("""
            SELECT 
                DATE_TRUNC('hour', created_at) as hour,
                COUNT(*) as cnt
            FROM predictions 
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY DATE_TRUNC('hour', created_at)
            ORDER BY hour
        """)
        stats["hourly_predictions_24h"] = [
            {"hour": r[0].isoformat(), "count": r[1]} for r in cur.fetchall()
        ]
        
        cur.close()
        conn.close()
        return stats
    except Exception as e:
        conn.close()
        return {"error": str(e)}

# ============================================================================
# MODEL ARCHITECTURE (must match training)
# ============================================================================

def get_model_class():
    """Define the model class inside a function to avoid import issues."""
    import torch
    import torch.nn as nn
    import torchvision.models as models
    
    class OrdinalRegressionResNet(nn.Module):
        def __init__(self, num_companies=None, company_embedding_dim=32, freeze_backbone=False):
            super(OrdinalRegressionResNet, self).__init__()
            
            self.use_company = num_companies is not None
            
            if self.use_company:
                self.company_embedding = nn.Embedding(num_companies, company_embedding_dim)
            
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.obverse_encoder = nn.Sequential(*list(resnet.children())[:-1])
            
            resnet2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.reverse_encoder = nn.Sequential(*list(resnet2.children())[:-1])
            
            self.feature_dim = 2048
            
            fusion_input_dim = self.feature_dim * 2
            if self.use_company:
                fusion_input_dim += company_embedding_dim
            
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            )
            
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
                nn.Sigmoid()
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
    
    return OrdinalRegressionResNet


# ============================================================================
# COIN GRADER CLASS
# ============================================================================

@app.cls(
    image=image,
    volumes={MODEL_DIR: model_volume},
    gpu="T4",  # Use T4 GPU for fast inference (change to "A10G" for faster)
    scaledown_window=300,  # Keep warm for 5 minutes
)
@modal.concurrent(max_inputs=20)  # Handle multiple concurrent requests per GPU container
class CoinGrader:
    """Modal class for coin grade prediction."""
    
    # Class-level state
    model = None
    transform = None
    company_to_idx = None
    use_company = False
    valid_grades = None
    num_steps = None
    encoding_type = "step_based"
    
    @modal.enter()
    def load_model(self):
        """Load model when container starts."""
        import torch
        from torchvision import transforms
        
        print("🔄 Loading coin grading model...")
        
        model_path = Path(MODEL_DIR) / "coin_ordinal_best.pth"
        
        if not model_path.exists():
            print(f"❌ Model not found at {model_path}")
            print("Please upload your model first with:")
            print("  modal volume put coin-grader-models coin_ordinal_best.pth")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract config from checkpoint
        self.use_company = checkpoint.get('use_company', False)
        self.company_to_idx = checkpoint.get('company_to_idx')
        self.encoding_type = checkpoint.get('encoding', 'step_based')
        
        if self.encoding_type == 'step_based':
            self.valid_grades = checkpoint.get('valid_grades', 
                [2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45,
                 50, 53, 55, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68])
            self.num_steps = len(self.valid_grades)
        
        # Build model
        OrdinalRegressionResNet = get_model_class()
        num_companies = len(self.company_to_idx) if self.use_company and self.company_to_idx else None
        
        self.model = OrdinalRegressionResNet(
            num_companies=num_companies,
            company_embedding_dim=32,
            freeze_backbone=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.device = device
        print(f"✅ Model loaded successfully on {device}")
        print(f"   Encoding: {self.encoding_type}")
        print(f"   Company conditioning: {self.use_company}")
    
    def preprocess_coin_image(self, pil_image, output_size=512):
        """Hough circle detection, circular mask, and centering on white background."""
        import numpy as np
        import cv2
        from PIL import Image
        
        img_rgb = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        height, width = img_bgr.shape[:2]
        min_dim = min(height, width)
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dim // 2,
            param1=50, param2=30,
            minRadius=int(min_dim * 0.2), maxRadius=int(min_dim * 0.5)
        )
        
        if circles is None:
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dim // 3,
                param1=100, param2=20,
                minRadius=int(min_dim * 0.15), maxRadius=int(min_dim * 0.55)
            )
        
        if circles is None:
            cx, cy = width // 2, height // 2
            radius = min_dim // 2 - 10
        else:
            circles = np.uint16(np.around(circles))
            cx, cy, radius = circles[0][0]
        
        # Create circular mask on original image first
        mask_radius = int(radius * 1.02)  # Slight padding
        mask_original = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        cv2.circle(mask_original, (int(cx), int(cy)), mask_radius, 255, -1)
        
        # Feather mask edges
        mask_original = cv2.GaussianBlur(mask_original, (7, 7), 0)
        
        # Blend coin onto white using mask
        white_original = np.ones_like(img_bgr) * 255
        mask_3ch = cv2.cvtColor(mask_original, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
        blended = (img_bgr.astype(float) * mask_3ch + white_original.astype(float) * (1 - mask_3ch)).astype(np.uint8)
        
        # Crop the masked region
        padding = int(radius * 0.05)
        crop_radius = radius + padding
        x1 = max(0, int(cx - crop_radius))
        y1 = max(0, int(cy - crop_radius))
        x2 = min(width, int(cx + crop_radius))
        y2 = min(height, int(cy + crop_radius))
        
        cropped = blended[y1:y2, x1:x2]
        
        white_bg = np.ones((output_size, output_size, 3), dtype=np.uint8) * 255
        
        crop_h, crop_w = cropped.shape[:2]
        scale = (output_size * 0.92) / max(crop_h, crop_w)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        x_offset = (output_size - new_w) // 2
        y_offset = (output_size - new_h) // 2
        white_bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        result_rgb = cv2.cvtColor(white_bg, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    
    def denormalize_grade(self, normalized):
        """Convert normalized prediction to Sheldon grade."""
        if self.encoding_type == 'step_based':
            step = normalized * (self.num_steps - 1)
            step_idx = int(round(step))
            step_idx = max(0, min(step_idx, self.num_steps - 1))
            return self.valid_grades[step_idx]
        else:
            return normalized * 69 + 1  # 1-70 scale
    
    def format_grade_name(self, sheldon_grade):
        """Convert Sheldon number to grade name."""
        grade = int(round(sheldon_grade))
        if grade >= 60:
            return f"MS{grade}"
        elif grade >= 50:
            return f"AU{grade}"
        elif grade >= 40:
            return f"XF{grade}"
        elif grade >= 20:
            return f"VF{grade}"
        elif grade >= 12:
            return f"F{grade}"
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
    
    @modal.method()
    def predict(self, obverse_bytes: bytes, reverse_bytes: bytes, company: str = None) -> dict:
        """Predict coin grade from image bytes."""
        import torch
        from PIL import Image
        
        # Load and preprocess images
        obverse_img = Image.open(io.BytesIO(obverse_bytes)).convert("RGB")
        reverse_img = Image.open(io.BytesIO(reverse_bytes)).convert("RGB")
        
        obverse_processed = self.preprocess_coin_image(obverse_img)
        reverse_processed = self.preprocess_coin_image(reverse_img)
        
        obverse_tensor = self.transform(obverse_processed).unsqueeze(0).to(self.device)
        reverse_tensor = self.transform(reverse_processed).unsqueeze(0).to(self.device)
        
        # Handle company conditioning
        company_idx_tensor = None
        company_used = None
        
        if self.use_company and self.company_to_idx:
            if company and company.upper() in self.company_to_idx:
                company_idx_tensor = torch.tensor([self.company_to_idx[company.upper()]]).to(self.device)
                company_used = company.upper()
            else:
                default = 'PCGS' if 'PCGS' in self.company_to_idx else list(self.company_to_idx.keys())[0]
                company_idx_tensor = torch.tensor([self.company_to_idx[default]]).to(self.device)
                company_used = f"{default} (default)"
        
        # Predict
        with torch.no_grad():
            prediction = self.model(obverse_tensor, reverse_tensor, company_idx_tensor)
            normalized_pred = prediction.item()
            sheldon_grade = self.denormalize_grade(normalized_pred)
            grade_name = self.format_grade_name(sheldon_grade)
        
        # Confidence
        if self.encoding_type == 'step_based':
            step_pos = normalized_pred * (self.num_steps - 1)
            confidence = 1.0 - abs(step_pos - round(step_pos))
            confidence = max(0.5, confidence)
        else:
            confidence = 0.85
        
        return {
            "prediction": grade_name,
            "sheldon_grade": int(sheldon_grade),
            "confidence": round(confidence * 100, 1),
            "raw_score": round(normalized_pred, 4),
            "company_used": company_used
        }
    
    @modal.method()
    def get_companies(self) -> list:
        """Return list of supported grading companies."""
        if self.company_to_idx:
            return list(self.company_to_idx.keys())
        return []
    
    @modal.method()
    def get_grades(self) -> list:
        """Return list of valid grades."""
        if self.valid_grades:
            return [{"sheldon": g, "name": self.format_grade_name(g)} for g in self.valid_grades]
        return []


# ============================================================================
# WEB ENDPOINTS (FastAPI)
# ============================================================================

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Optional
import secrets

web_app = FastAPI(title="Numisking Coin Grader")

# Static files are mounted at /app/static in the container
CONTAINER_STATIC_DIR = "/app/static"

# HTTP Basic Auth for admin endpoints
security = HTTPBasic()


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify admin credentials for protected endpoints."""
    import os
    
    # Read credentials from Modal secrets (environment variables)
    admin_username = os.environ.get("USERNAME", "")
    admin_password = os.environ.get("PASSWORD", "")
    
    if not admin_username or not admin_password:
        raise HTTPException(
            status_code=500,
            detail="Admin credentials not configured",
        )
    
    correct_username = secrets.compare_digest(credentials.username, admin_username)
    correct_password = secrets.compare_digest(credentials.password, admin_password)

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies."""
    # Check X-Forwarded-For header (set by proxies/load balancers)
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    # Check X-Real-IP header
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip
    # Fall back to direct client
    return request.client.host if request.client else "unknown"


@web_app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    init_database()


@web_app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main Numisking UI."""
    index_path = f"{CONTAINER_STATIC_DIR}/index.html"
    try:
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Loading...</h1><p>If this persists, redeploy with: modal deploy modal_app.py</p>",
            status_code=200
        )


@web_app.post("/predict")
async def predict(
    request: Request,
    obverse: UploadFile = File(...),
    reverse: UploadFile = File(...),
    company: Optional[str] = Form(None),
    async_mode: bool = Form(False, description="Return job_id for polling instead of waiting"),
    save_images: bool = Form(True, description="Save images for review (default True)")
):
    """Predict coin grade from uploaded images.
    
    If async_mode=True, returns immediately with a job_id that can be polled at /predict/{job_id}.
    If async_mode=False (default), waits for the result (uses async I/O, doesn't block other requests).
    """
    start_time = time.time()
    
    # Get client info for logging
    client_ip = get_client_ip(request)
    user_agent = request.headers.get("user-agent", "")
    
    grader = CoinGrader()
    
    obverse_bytes = await obverse.read()
    reverse_bytes = await reverse.read()
    
    if async_mode:
        # Fire-and-forget: return job_id immediately for polling
        function_call = grader.predict.spawn(obverse_bytes, reverse_bytes, company)
        job_id = function_call.object_id
        
        # Log async request (result will be logged when polled)
        prediction_id = log_prediction(
            ip=client_ip,
            user_agent=user_agent,
            result={"status": "async"},
            processing_time_ms=int((time.time() - start_time) * 1000),
            async_mode=True,
            job_id=job_id,
            has_images=save_images
        )
        
        # Save images in background
        if save_images and prediction_id:
            save_images_worker.spawn(prediction_id, obverse_bytes, reverse_bytes)
        
        return {"job_id": job_id, "status": "processing"}
    else:
        # Async wait: doesn't block other requests thanks to .remote.aio()
        result = await grader.predict.remote.aio(obverse_bytes, reverse_bytes, company)
        
        # Log the prediction
        processing_time_ms = int((time.time() - start_time) * 1000)
        prediction_id = log_prediction(
            ip=client_ip,
            user_agent=user_agent,
            result=result,
            processing_time_ms=processing_time_ms,
            has_images=save_images
        )
        
        # Save images in background (don't wait for it)
        if save_images and prediction_id:
            save_images_worker.spawn(prediction_id, obverse_bytes, reverse_bytes)
        
        return result


@web_app.get("/predict/{job_id}")
async def get_prediction_result(job_id: str):
    """Poll for prediction result by job_id.
    
    Returns status: 'processing', 'completed', or 'failed'.
    When completed, includes the full prediction result.
    """
    from modal.functions import FunctionCall
    
    try:
        function_call = FunctionCall.from_id(job_id)
        
        try:
            # Try to get result without blocking (timeout=0)
            result = function_call.get(timeout=0)
            return {"status": "completed", "result": result}
        except TimeoutError:
            return {"status": "processing", "job_id": job_id}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def detect_coins_contour_modal(img, cv2, np):
    """Detect coins using contour detection - more robust than Hough circles."""
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 31, 10
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    coin_candidates = []
    min_area = (min(height, width) * 0.1) ** 2
    max_area = (min(height, width) * 0.6) ** 2
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        if circularity < 0.5:
            continue
        
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        coin_candidates.append({
            'cx': int(cx), 'cy': int(cy), 'radius': int(radius),
            'area': area, 'circularity': circularity
        })
    
    return coin_candidates


def detect_coins_color_modal(img, cv2, np):
    """Detect coins using color segmentation for metallic objects."""
    height, width = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mask_gold = cv2.inRange(hsv, (5, 30, 80), (35, 255, 255))
    mask_silver = cv2.inRange(hsv, (0, 0, 100), (180, 50, 255))
    mask_brown = cv2.inRange(hsv, (0, 20, 50), (20, 150, 200))
    
    mask = cv2.bitwise_or(mask_gold, mask_silver)
    mask = cv2.bitwise_or(mask, mask_brown)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    coin_candidates = []
    min_area = (min(height, width) * 0.1) ** 2
    max_area = (min(height, width) * 0.6) ** 2
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        if circularity < 0.4:
            continue
        
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        coin_candidates.append({
            'cx': int(cx), 'cy': int(cy), 'radius': int(radius),
            'area': area, 'circularity': circularity
        })
    
    return coin_candidates


def split_image_simple_modal(img):
    """Simple fallback: split image into left and right halves."""
    height, width = img.shape[:2]
    
    margin_x = int(width * 0.02)
    margin_y = int(height * 0.05)
    half_width = width // 2
    
    left_cx = (margin_x + half_width - margin_x) // 2
    right_cx = half_width + (margin_x + half_width - margin_x) // 2
    cy = height // 2
    radius = min(half_width - 2 * margin_x, height - 2 * margin_y) // 2
    
    return [
        {'cx': left_cx, 'cy': cy, 'radius': radius, 'area': 0, 'circularity': 1.0},
        {'cx': right_cx, 'cy': cy, 'radius': radius, 'area': 0, 'circularity': 1.0}
    ]


# ============================================================================
# IMAGE STORAGE WORKER - Save prediction images to volume
# ============================================================================

@app.function(
    image=image,
    volumes={IMAGES_DIR: images_volume},
    secrets=[db_secret],
    cpu=0.5,
    memory=256,
    scaledown_window=60,
)
def save_images_worker(prediction_id: int, obverse_bytes: bytes, reverse_bytes: bytes):
    """Save prediction images to the images volume."""
    import os
    
    try:
        # Create directory for this prediction
        pred_dir = f"{IMAGES_DIR}/{prediction_id}"
        os.makedirs(pred_dir, exist_ok=True)
        
        # Save images
        with open(f"{pred_dir}/obverse.jpg", "wb") as f:
            f.write(obverse_bytes)
        
        with open(f"{pred_dir}/reverse.jpg", "wb") as f:
            f.write(reverse_bytes)
        
        # Commit changes to volume
        images_volume.commit()
        
        print(f"✅ Saved images for prediction {prediction_id}")
        return True
    except Exception as e:
        print(f"❌ Error saving images for prediction {prediction_id}: {e}")
        return False


@app.function(
    image=image,
    volumes={IMAGES_DIR: images_volume},
    secrets=[db_secret],
    cpu=0.5,
    memory=512,
    scaledown_window=60,
)
def save_split_images_worker(split_id: int, original_bytes: bytes, obverse_b64: str, reverse_b64: str):
    """Save split request images (original + detected coins) to the images volume."""
    import os
    import base64
    
    try:
        # Create directory for this split request
        split_dir = f"{IMAGES_DIR}/splits/{split_id}"
        os.makedirs(split_dir, exist_ok=True)
        
        # Save original TrueView image
        with open(f"{split_dir}/original.jpg", "wb") as f:
            f.write(original_bytes)
        
        # Decode and save obverse
        obverse_bytes = base64.b64decode(obverse_b64)
        with open(f"{split_dir}/obverse.jpg", "wb") as f:
            f.write(obverse_bytes)
        
        # Decode and save reverse
        reverse_bytes = base64.b64decode(reverse_b64)
        with open(f"{split_dir}/reverse.jpg", "wb") as f:
            f.write(reverse_bytes)
        
        # Commit changes to volume
        images_volume.commit()
        
        print(f"✅ Saved split images for split {split_id}")
        return True
    except Exception as e:
        print(f"❌ Error saving split images for split {split_id}: {e}")
        return False


# ============================================================================
# IMAGE SPLITTER - Dedicated CPU function for splitting combined images
# ============================================================================

@app.function(
    image=image,
    cpu=2,  # 2 CPU cores for image processing
    memory=1024,  # 1GB RAM
    scaledown_window=120,  # Keep warm for 2 minutes
)
@modal.concurrent(max_inputs=50)  # Handle many concurrent split requests
def split_image_worker(image_bytes: bytes) -> dict:
    """
    CPU worker to split a combined TrueView/slab image into obverse and reverse.
    
    Runs on dedicated CPU containers, auto-scales independently from GPU workers.
    """
    import cv2
    import numpy as np
    import base64
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Could not read image"}
    
    height, width = img.shape[:2]
    
    # Try multiple detection methods
    coins = []
    detection_method = "none"
    
    # Method 1: Contour detection
    coins = detect_coins_contour_modal(img, cv2, np)
    if len(coins) >= 2:
        detection_method = "contour"
    
    # Method 2: Color segmentation
    if len(coins) < 2:
        color_coins = detect_coins_color_modal(img, cv2, np)
        if len(color_coins) >= 2:
            coins = color_coins
            detection_method = "color"
        elif len(color_coins) > len(coins):
            coins = color_coins
    
    # Method 3: Simple geometric split (fallback)
    if len(coins) < 2:
        coins = split_image_simple_modal(img)
        detection_method = "geometric"
    
    # Sort by x coordinate
    coins = sorted(coins, key=lambda c: c['cx'])
    
    if len(coins) > 2 and detection_method != "geometric":
        coins = sorted(coins, key=lambda c: c['area'], reverse=True)[:2]
        coins = sorted(coins, key=lambda c: c['cx'])
    
    obverse_coin = coins[0]
    reverse_coin = coins[1] if len(coins) > 1 else coins[0]
    
    def extract_coin(coin_info, img, output_size=512):
        cx, cy, r = coin_info['cx'], coin_info['cy'], coin_info['radius']
        
        # Add small padding to radius
        padding = int(r * 0.05)
        mask_r = r + padding
        
        # Create circular mask on original image
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (cx, cy), mask_r, 255, -1)
        
        # Apply slight feathering to mask edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Create white background same size as original
        white_img = np.ones_like(img) * 255
        
        # Blend coin onto white using mask
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255
        result = (img.astype(float) * mask_3ch + white_img.astype(float) * (1 - mask_3ch)).astype(np.uint8)
        
        # Crop to bounding box of the coin
        crop_r = mask_r + 10
        x1 = max(0, cx - crop_r)
        y1 = max(0, cy - crop_r)
        x2 = min(img.shape[1], cx + crop_r)
        y2 = min(img.shape[0], cy + crop_r)
        
        cropped = result[y1:y2, x1:x2]
        white_bg = np.ones((output_size, output_size, 3), dtype=np.uint8) * 255
        
        crop_h, crop_w = cropped.shape[:2]
        if crop_h == 0 or crop_w == 0:
            return white_bg
        
        scale = (output_size * 0.92) / max(crop_h, crop_w)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        x_offset = (output_size - new_w) // 2
        y_offset = (output_size - new_h) // 2
        white_bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return white_bg
    
    obverse_img = extract_coin(obverse_coin, img)
    reverse_img = extract_coin(reverse_coin, img)
    
    _, obverse_encoded = cv2.imencode('.jpg', obverse_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    _, reverse_encoded = cv2.imencode('.jpg', reverse_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    
    obverse_b64 = base64.b64encode(obverse_encoded).decode('utf-8')
    reverse_b64 = base64.b64encode(reverse_encoded).decode('utf-8')
    
    return {
        "obverse": obverse_b64,
        "reverse": reverse_b64,
        "detection_method": detection_method,
        "coins_found": len(coins)
    }


@web_app.post("/split-combined")
async def split_combined(
    request: Request, 
    image: UploadFile = File(...),
    save_images: bool = Form(True, description="Save original and split images (default True)")
):
    """
    Split a combined TrueView/slab image into obverse and reverse.

    Offloads CPU-intensive work to a dedicated worker that auto-scales.
    Saves the original TrueView image along with the detected coins.
    """
    start_time = time.time()

    # Get client info for logging
    client_ip = get_client_ip(request)
    user_agent = request.headers.get("user-agent", "")

    image_bytes = await image.read()

    # Offload to CPU worker - non-blocking async call
    result = await split_image_worker.remote.aio(image_bytes)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # Log the split request
    processing_time_ms = int((time.time() - start_time) * 1000)
    split_id = log_split_request(
        ip=client_ip,
        user_agent=user_agent,
        result=result,
        processing_time_ms=processing_time_ms,
        has_images=save_images
    )

    # Save original and split images in background
    if save_images and split_id:
        save_split_images_worker.spawn(
            split_id, 
            image_bytes, 
            result["obverse"],  # base64 
            result["reverse"]   # base64
        )
        result["split_id"] = split_id

    return result


@web_app.get("/companies")
async def get_companies():
    """Get list of supported grading companies."""
    grader = CoinGrader()
    companies = grader.get_companies.remote()
    return {"companies": companies}


@web_app.get("/grades")
async def get_grades():
    """Get list of valid grades."""
    grader = CoinGrader()
    grades = grader.get_grades.remote()
    return {"grades": grades}


@web_app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@web_app.get("/admin/stats")
async def admin_stats(request: Request, username: str = Depends(verify_admin)):
    """
    Get analytics statistics. (Protected - requires admin login)
    
    Returns usage stats including:
    - Total predictions and splits
    - Predictions today / last 7 days
    - Unique IPs
    - Grade distribution
    - Top users by IP
    - Hourly predictions (last 24h)
    """
    stats = get_analytics_stats()
    
    if "error" in stats:
        return JSONResponse(
            content={"error": stats["error"], "message": "Database not available"},
            status_code=503
        )
    
    return stats


@web_app.get("/admin/recent")
async def admin_recent(limit: int = 50, offset: int = 0, username: str = Depends(verify_admin)):
    """
    Get recent predictions. (Protected - requires admin login)
    
    Returns the most recent predictions with details.
    """
    conn = get_db_connection()
    if not conn:
        return JSONResponse(
            content={"error": "Database not available"},
            status_code=503
        )
    
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                id, created_at, ip_address, predicted_grade, 
                sheldon_grade, confidence, company_used, 
                processing_time_ms, async_mode, has_images
            FROM predictions
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """, (min(limit, 500), offset))
        
        columns = ['id', 'created_at', 'ip_address', 'predicted_grade', 
                   'sheldon_grade', 'confidence', 'company_used', 
                   'processing_time_ms', 'async_mode', 'has_images']
        
        predictions = []
        for row in cur.fetchall():
            pred = dict(zip(columns, row))
            pred['created_at'] = pred['created_at'].isoformat() if pred['created_at'] else None
            predictions.append(pred)
        
        cur.close()
        conn.close()
        return {"predictions": predictions}
    except Exception as e:
        conn.close()
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@web_app.get("/admin/trends")
async def admin_trends(period: str = "7d", username: str = Depends(verify_admin)):
    """
    Get prediction trends for charting. (Protected - requires admin login)
    
    Periods: 24h (hourly), 7d (daily), 30d (daily)
    """
    conn = get_db_connection()
    if not conn:
        return JSONResponse(
            content={"error": "Database not available"},
            status_code=503
        )
    
    try:
        cur = conn.cursor()
        
        if period == "24h":
            cur.execute("""
                SELECT 
                    DATE_TRUNC('hour', created_at) as period,
                    COUNT(*) as count
                FROM predictions 
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                GROUP BY DATE_TRUNC('hour', created_at)
                ORDER BY period
            """)
            trends = [{"label": r[0].strftime("%H:%M"), "count": r[1]} for r in cur.fetchall()]
        elif period == "30d":
            cur.execute("""
                SELECT 
                    DATE_TRUNC('day', created_at) as period,
                    COUNT(*) as count
                FROM predictions 
                WHERE created_at >= NOW() - INTERVAL '30 days'
                GROUP BY DATE_TRUNC('day', created_at)
                ORDER BY period
            """)
            trends = [{"label": r[0].strftime("%b %d"), "count": r[1]} for r in cur.fetchall()]
        else:  # 7d default
            cur.execute("""
                SELECT 
                    DATE_TRUNC('day', created_at) as period,
                    COUNT(*) as count
                FROM predictions 
                WHERE created_at >= NOW() - INTERVAL '7 days'
                GROUP BY DATE_TRUNC('day', created_at)
                ORDER BY period
            """)
            trends = [{"label": r[0].strftime("%a"), "count": r[1]} for r in cur.fetchall()]
        
        cur.close()
        conn.close()
        return {"trend": trends, "period": period}
    except Exception as e:
        conn.close()
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@web_app.get("/admin/images/{prediction_id}/{side}")
async def get_prediction_image_endpoint(prediction_id: int, side: str, username: str = Depends(verify_admin)):
    """
    Get a prediction image (obverse or reverse). (Protected - requires admin login)
    """
    from fastapi.responses import Response
    import os

    if side not in ["obverse", "reverse"]:
        raise HTTPException(status_code=400, detail="Side must be 'obverse' or 'reverse'")

    # Reload volume to see latest images from other containers
    try:
        images_volume.reload()
    except Exception as e:
        print(f"Volume reload warning: {e}")

    # Try to get image from volume
    image_path = f"{IMAGES_DIR}/{prediction_id}/{side}.jpg"

    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            return Response(
                content=image_bytes,
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=3600"}  # Cache for 1 hour
            )
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@web_app.get("/admin/splits/{split_id}/{image_type}")
async def get_split_image_endpoint(split_id: int, image_type: str, username: str = Depends(verify_admin)):
    """
    Get a split request image (original, obverse, or reverse). (Protected - requires admin login)
    
    - original: The original TrueView/combined image
    - obverse: The detected obverse coin
    - reverse: The detected reverse coin
    """
    from fastapi.responses import Response
    import os

    if image_type not in ["original", "obverse", "reverse"]:
        raise HTTPException(status_code=400, detail="Image type must be 'original', 'obverse', or 'reverse'")

    # Reload volume to see latest images from other containers
    try:
        images_volume.reload()
    except Exception as e:
        print(f"Volume reload warning: {e}")

    # Try to get image from volume
    image_path = f"{IMAGES_DIR}/splits/{split_id}/{image_type}.jpg"

    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            return Response(
                content=image_bytes,
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=3600"}  # Cache for 1 hour
            )
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@web_app.get("/admin", response_class=HTMLResponse)
async def admin_page(username: str = Depends(verify_admin)):
    """Serve the admin dashboard. (Protected - requires admin login)"""
    admin_path = f"{CONTAINER_STATIC_DIR}/admin.html"
    try:
        with open(admin_path, "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Admin dashboard not found</h1><p>Redeploy with: modal deploy modal_app.py</p>",
            status_code=404
        )


@web_app.get("/static/{filename:path}")
async def serve_static(filename: str):
    """Serve static files (CSS, JS, images)."""
    import os
    file_path = f"{CONTAINER_STATIC_DIR}/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found"}, 404


# Mount the FastAPI app
@app.function(
    image=image,
    volumes={
        MODEL_DIR: model_volume,
        IMAGES_DIR: images_volume,  # For serving prediction images
    },
    secrets=[db_secret, admin_secret],  # Include database and admin credentials
    allow_concurrent_inputs=100
)
@modal.asgi_app()
def fastapi_app():
    return web_app


# ============================================================================
# CLI COMMANDS
# ============================================================================

@app.local_entrypoint()
def main():
    """Local entrypoint for testing."""
    print("🪙 Numisking Coin Grader")
    print("=" * 50)
    print("\nTo deploy:")
    print("  modal deploy modal_app.py")
    print("\nTo run locally:")
    print("  modal serve modal_app.py")
    print("\nTo upload your model:")
    print("  modal volume put coin-grader-models /path/to/coin_ordinal_best.pth")
    print("\nAfter deployment, your app will be at:")
    print("  https://YOUR_USERNAME--numisking-coin-grader-fastapi-app.modal.run")

