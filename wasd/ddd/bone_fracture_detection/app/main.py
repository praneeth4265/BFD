"""
FastAPI backend for bone fracture detection with ONNX Runtime inference.
Provides REST API endpoints for image upload and prediction.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import onnxruntime as ort
import numpy as np
import cv2
import io
import json
import base64
from PIL import Image
from typing import Dict, Any, List, Optional
import logging
import time
from pathlib import Path
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bone Fracture Detection API",
    description="Medical AI API for bone fracture detection in X-ray images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))


class BoneFractureInference:
    """ONNX-based inference engine for bone fracture detection."""
    
    def __init__(self, model_path: str, metadata_path: str = None):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to ONNX model
            metadata_path: Path to metadata JSON file
        """
        try:
            # Initialize ONNX Runtime session
            providers = ['CPUExecutionProvider']
            if ort.get_available_providers():
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
            logger.info(f"Model loaded successfully with providers: {providers}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
        
        # Load metadata
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            # Default metadata
            self.metadata = {
                'input_shape': [1, 3, 224, 224],
                'preprocessing': {
                    'resize': [224, 224],
                    'normalize_mean': [0.485, 0.456, 0.406],
                    'normalize_std': [0.229, 0.224, 0.225],
                    'apply_clahe': True
                },
                'output_classes': ['No Fracture', 'Fracture']
            }
        
        logger.info("Inference engine initialized successfully")
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement to image."""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            # Merge back
            lab[:, :, 0] = l_channel
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess input image for inference.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image tensor
        """
        # Get preprocessing parameters
        target_size = tuple(self.metadata['preprocessing']['resize'])
        mean = np.array(self.metadata['preprocessing']['normalize_mean'])
        std = np.array(self.metadata['preprocessing']['normalize_std'])
        apply_clahe = self.metadata['preprocessing'].get('apply_clahe', True)
        
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            # Convert grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Apply CLAHE enhancement
        if apply_clahe:
            image = self.apply_clahe(image)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        image = (image - mean) / std
        
        # Add batch dimension and transpose to NCHW
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = np.transpose(image, (0, 3, 1, 2))  # NHWC to NCHW
        
        return image
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Make prediction on input image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Prediction results
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: processed_image})
            logits = outputs[0]
            
            # Apply softmax to get probabilities
            probabilities = self._softmax(logits)
            
            # Get prediction
            predicted_class = int(np.argmax(probabilities, axis=1)[0])
            confidence = float(probabilities[0][predicted_class])
            
            inference_time = time.time() - start_time
            
            return {
                'success': True,
                'predicted_class': predicted_class,
                'predicted_label': self.metadata['output_classes'][predicted_class],
                'confidence': confidence,
                'probabilities': probabilities[0].tolist(),
                'class_names': self.metadata['output_classes'],
                'inference_time_ms': inference_time * 1000,
                'model_info': {
                    'input_shape': self.input_shape,
                    'providers': [provider.replace('ExecutionProvider', '') for provider in self.session.get_providers()]
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'predicted_class': None,
                'predicted_label': None,
                'confidence': 0.0
            }
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Global inference engine
inference_engine: Optional[BoneFractureInference] = None


def get_inference_engine() -> BoneFractureInference:
    """Dependency to get inference engine."""
    global inference_engine
    if inference_engine is None:
        # Look for model files
        model_dir = Path(__file__).parent.parent / "models"
        
        # Try to find ONNX model
        onnx_files = list(model_dir.glob("*.onnx"))
        if not onnx_files:
            raise HTTPException(
                status_code=500,
                detail="No ONNX model found. Please place an ONNX model in the models directory."
            )
        
        model_path = str(onnx_files[0])  # Use first ONNX file found
        
        # Look for metadata file
        metadata_files = list(model_dir.glob("*metadata.json"))
        metadata_path = str(metadata_files[0]) if metadata_files else None
        
        try:
            inference_engine = BoneFractureInference(model_path, metadata_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize inference engine: {e}"
            )
    
    return inference_engine


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Bone Fracture Detection API...")
    try:
        # Initialize inference engine
        get_inference_engine()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "message": "Bone Fracture Detection API is running"
    }


@app.get("/model/info")
async def model_info(engine: BoneFractureInference = Depends(get_inference_engine)):
    """Get model information."""
    return {
        "input_shape": engine.input_shape,
        "output_classes": engine.metadata['output_classes'],
        "preprocessing": engine.metadata['preprocessing'],
        "providers": [provider.replace('ExecutionProvider', '') for provider in engine.session.get_providers()]
    }


@app.post("/predict")
async def predict_fracture(
    file: UploadFile = File(...),
    engine: BoneFractureInference = Depends(get_inference_engine)
):
    """
    Predict bone fracture from uploaded X-ray image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    # Validate file size (10MB limit)
    max_size = 10 * 1024 * 1024  # 10MB
    file_size = 0
    
    try:
        # Read file content
        contents = await file.read()
        file_size = len(contents)
        
        if file_size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds limit (10MB)"
            )
        
        # Convert to numpy array
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Make prediction
        result = engine.predict(image_array)
        
        # Add file info to result
        result['file_info'] = {
            'filename': file.filename,
            'size_bytes': file_size,
            'content_type': file.content_type
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    engine: BoneFractureInference = Depends(get_inference_engine)
):
    """
    Predict bone fractures for multiple images.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        List of prediction results
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Batch size cannot exceed 10 images"
        )
    
    results = []
    
    for file in files:
        try:
            # Validate file
            if not file.content_type.startswith('image/'):
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': 'File must be an image'
                })
                continue
            
            # Read and process file
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            image_array = np.array(image)
            
            # Make prediction
            result = engine.predict(image_array)
            result['filename'] = file.filename
            results.append(result)
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'success': False,
                'error': str(e)
            })
    
    return {'results': results}


@app.get("/examples")
async def get_examples():
    """Get example images for testing."""
    examples_dir = static_dir / "examples"
    if not examples_dir.exists():
        return {"examples": []}
    
    examples = []
    for img_file in examples_dir.glob("*.jpg"):
        examples.append({
            'filename': img_file.name,
            'url': f"/static/examples/{img_file.name}"
        })
    
    return {"examples": examples}


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )