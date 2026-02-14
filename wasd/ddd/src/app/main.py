"""
FastAPI backend for 4-model ensemble bone fracture classification.

Models: ConvNeXt V2 Base, EfficientNetV2-S, MaxViT-Tiny, Swin Transformer
Classes: comminuted_fracture, no_fracture, simple_fracture

Endpoints:
    POST /api/predict            -- single image prediction
    POST /api/predict/batch      -- batch prediction (up to 16 images)
    POST /api/predict/heatmap    -- prediction + Grad-CAM heatmaps
    GET  /api/model/info         -- model architecture and weight details
    GET  /api/health             -- health check + GPU status
    GET  /                       -- serves the web frontend

Run:
    cd <project_root>
    python src/app/main.py
"""

from __future__ import annotations

import io
import sys
import time
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Resolve project root so imports work regardless of cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/app/main.py -> project root
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ensemble"))

from ensemble_model import EnsembleModel, EnsemblePrediction, CLASSES  # noqa: E402

# Grad-CAM generator (imported lazily after ensemble is loaded)
gradcam_generator = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("bfd-api")

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class ClassProbability(BaseModel):
    class_name: str
    probability: float


class IndividualModelResult(BaseModel):
    model_name: str
    prediction: str
    weight: float
    active: bool


class PredictionResult(BaseModel):
    prediction: str
    prediction_display: str
    confidence: float
    probabilities: List[ClassProbability]
    individual_models: List[IndividualModelResult]
    agreement_count: int
    inference_time_ms: float


class BatchPredictionResult(BaseModel):
    results: List[PredictionResult]
    total_images: int
    total_inference_time_ms: float
    avg_inference_time_ms: float


class ModelDetail(BaseModel):
    name: str
    architecture: str
    parameters: str
    weight: float
    active: bool


class ModelInfo(BaseModel):
    project: str
    version: str
    ensemble_type: str
    num_models: int
    active_models: int
    classes: List[str]
    num_classes: int
    input_size: int
    weights: dict
    models: List[ModelDetail]
    device: str
    test_accuracy: str
    test_samples: int
    optimization_method: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    num_models: int
    device: str
    gpu_name: Optional[str] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    uptime_seconds: float


class HeatmapModelResult(BaseModel):
    name: str
    heatmap_b64: Optional[str] = None
    active: bool
    weight: float


class HeatmapResponse(BaseModel):
    prediction: str
    prediction_display: str
    confidence: float
    probabilities: List[ClassProbability]
    individual_models: List[IndividualModelResult]
    agreement_count: int
    inference_time_ms: float
    per_model_heatmaps: List[HeatmapModelResult]
    ensemble_heatmap_b64: str
    original_image_b64: str


# ---------------------------------------------------------------------------
# Display name mapping
# ---------------------------------------------------------------------------
DISPLAY_NAMES = {
    "comminuted_fracture": "Comminuted Fracture",
    "no_fracture": "No Fracture",
    "simple_fracture": "Simple Fracture",
}

ARCHITECTURE_MAP = {
    "ConvNeXt V2": "Modern CNN (Hierarchical Features)",
    "EfficientNetV2-S": "Efficient CNN (Compound Scaling)",
    "MaxViT-Tiny": "Hybrid CNN-Transformer (Multi-Axis Attention)",
    "Swin Transformer": "Vision Transformer (Shifted Windows)",
}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bone Fracture Detection API",
    description=(
        "4-model ensemble for classifying bone X-rays into comminuted fracture, "
        "simple fracture, or no fracture. "
        "Optimized weights achieve 100 percent accuracy on 3,082 test images."
    ),
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
ensemble: Optional[EnsembleModel] = None
START_TIME: float = time.time()

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def load_models():
    global ensemble, gradcam_generator
    logger.info("Loading 4-model ensemble...")
    t0 = time.time()

    import os
    os.chdir(PROJECT_ROOT)

    ensemble = EnsembleModel()
    elapsed = time.time() - t0
    logger.info("Ensemble loaded on %s in %.1fs", ensemble.device, elapsed)

    # Initialize Grad-CAM generator
    try:
        from gradcam_heatmap import GradCAMGenerator
        gradcam_generator = GradCAMGenerator(ensemble)
        logger.info("Grad-CAM generator initialized successfully")
    except Exception as e:
        logger.warning("Grad-CAM init failed (heatmaps will be unavailable): %s", e)
        gradcam_generator = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _read_image(file: UploadFile) -> Image.Image:
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Unsupported type: " + str(file.content_type))
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large (max 20 MB).")
    try:
        # Convert to grayscale then back to 3-channel to eliminate
        # color-channel artifacts that bias the model (dataset has
        # no_fracture=grayscale PNG, fracture=color JPG).
        return Image.open(io.BytesIO(data)).convert("L").convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image.")


def _to_response(pred: EnsemblePrediction, inference_ms: float) -> PredictionResult:
    individual_models = []
    for i, name in enumerate(ensemble.model_names):
        individual_models.append(IndividualModelResult(
            model_name=name,
            prediction=DISPLAY_NAMES.get(
                pred.individual_predictions[name],
                pred.individual_predictions[name],
            ),
            weight=ensemble.weights[i],
            active=ensemble.weights[i] > 0,
        ))

    return PredictionResult(
        prediction=pred.prediction,
        prediction_display=DISPLAY_NAMES.get(pred.prediction, pred.prediction),
        confidence=round(pred.confidence, 6),
        probabilities=[
            ClassProbability(
                class_name=DISPLAY_NAMES.get(cls, cls),
                probability=round(prob, 6),
            )
            for cls, prob in pred.probabilities.items()
        ],
        individual_models=individual_models,
        agreement_count=pred.agreement_count,
        inference_time_ms=round(inference_ms, 2),
    )


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_frontend():
    index = Path(__file__).parent / "templates" / "index.html"
    return FileResponse(str(index), media_type="text/html")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health():
    gpu_name = gpu_mem_used = gpu_mem_total = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_used = round(torch.cuda.memory_allocated(0) / 1024**2, 1)
        gpu_mem_total = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**2, 1
        )

    return HealthResponse(
        status="healthy" if ensemble else "loading",
        models_loaded=ensemble is not None,
        num_models=len(ensemble.models) if ensemble else 0,
        device=ensemble.device if ensemble else "N/A",
        gpu_name=gpu_name,
        gpu_memory_used_mb=gpu_mem_used,
        gpu_memory_total_mb=gpu_mem_total,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.get("/api/model/info", response_model=ModelInfo, tags=["System"])
async def model_info():
    if not ensemble:
        raise HTTPException(503, "Models not loaded yet.")

    details = []
    for i, name in enumerate(ensemble.model_names):
        params = sum(p.numel() for p in ensemble.models[i].parameters())
        details.append(ModelDetail(
            name=name,
            architecture=ARCHITECTURE_MAP.get(name, "Unknown"),
            parameters=str(round(params / 1e6, 1)) + "M",
            weight=ensemble.weights[i],
            active=ensemble.weights[i] > 0,
        ))

    return ModelInfo(
        project="BFD -- Bone Fracture Detection",
        version="2.0.0",
        ensemble_type="Soft Voting (Weighted Average)",
        num_models=len(ensemble.models),
        active_models=sum(1 for w in ensemble.weights if w > 0),
        classes=[DISPLAY_NAMES.get(c, c) for c in CLASSES],
        num_classes=len(CLASSES),
        input_size=224,
        weights=dict(zip(ensemble.model_names, ensemble.weights)),
        models=details,
        device=ensemble.device,
        test_accuracy="100.00% (540 images, clean original dataset)",
        test_samples=540,
        optimization_method="Grid search — 1,771 weight combinations (retrained on clean data)",
    )


@app.post("/api/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """Classify a single bone X-ray image."""
    if not ensemble:
        raise HTTPException(503, "Models not loaded yet.")

    img = await _read_image(file)
    tensor = TRANSFORM(img).unsqueeze(0).to(ensemble.device)

    t0 = time.time()
    results = ensemble.predict(tensor)
    inference_ms = (time.time() - t0) * 1000

    logger.info(
        "Predict: %s -> %s (%.4f) [%.1fms]",
        file.filename,
        results[0].prediction,
        results[0].confidence,
        inference_ms,
    )
    return _to_response(results[0], inference_ms)


@app.post("/api/predict/batch", response_model=BatchPredictionResult, tags=["Prediction"])
async def predict_batch(
    files: List[UploadFile] = File(...),
    batch_size: int = Query(default=16, ge=1, le=32),
):
    """Classify up to 16 bone X-ray images at once."""
    if not ensemble:
        raise HTTPException(503, "Models not loaded yet.")
    if len(files) > 16:
        raise HTTPException(400, "Maximum 16 images per batch.")

    images = [TRANSFORM(await _read_image(f)) for f in files]
    batch_tensor = torch.stack(images).to(ensemble.device)

    t0 = time.time()
    all_preds = []
    for i in range(0, len(images), batch_size):
        all_preds.extend(ensemble.predict(batch_tensor[i:i + batch_size]))
    total_ms = (time.time() - t0) * 1000

    return BatchPredictionResult(
        results=[_to_response(p, total_ms / len(all_preds)) for p in all_preds],
        total_images=len(files),
        total_inference_time_ms=round(total_ms, 2),
        avg_inference_time_ms=round(total_ms / len(files), 2),
    )


@app.post("/api/predict/heatmap", response_model=HeatmapResponse, tags=["Prediction"])
async def predict_heatmap(file: UploadFile = File(...)):
    """Classify a bone X-ray and return Grad-CAM heatmaps for each model + ensemble."""
    if not ensemble:
        raise HTTPException(503, "Models not loaded yet.")
    if not gradcam_generator:
        raise HTTPException(503, "Grad-CAM generator not available. Check server logs.")

    img = await _read_image(file)
    original_np = np.array(img.resize((224, 224)))
    tensor = TRANSFORM(img).unsqueeze(0).to(ensemble.device)

    t0 = time.time()
    # First get prediction for the standard response
    results = ensemble.predict(tensor)
    pred = results[0]

    # Generate Grad-CAM heatmaps
    heatmap_data = gradcam_generator.generate(tensor, original_np)
    inference_ms = (time.time() - t0) * 1000

    # Build per-model heatmap results
    per_model_heatmaps = [
        HeatmapModelResult(
            name=m["name"],
            heatmap_b64=m["heatmap_b64"],
            active=m["active"],
            weight=m["weight"],
        )
        for m in heatmap_data["per_model"]
    ]

    # Build standard prediction fields
    std_response = _to_response(pred, inference_ms)

    # Encode original image as base64 for frontend display
    import base64 as b64
    buf = io.BytesIO()
    img.resize((224, 224)).save(buf, format="PNG")
    original_b64 = b64.b64encode(buf.getvalue()).decode("ascii")

    logger.info(
        "Heatmap: %s -> %s (%.4f) [%.1fms]",
        file.filename,
        pred.prediction,
        pred.confidence,
        inference_ms,
    )

    return HeatmapResponse(
        prediction=std_response.prediction,
        prediction_display=std_response.prediction_display,
        confidence=std_response.confidence,
        probabilities=std_response.probabilities,
        individual_models=std_response.individual_models,
        agreement_count=std_response.agreement_count,
        inference_time_ms=std_response.inference_time_ms,
        per_model_heatmaps=per_model_heatmaps,
        ensemble_heatmap_b64=heatmap_data["ensemble_heatmap_b64"],
        original_image_b64=original_b64,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
