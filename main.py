"""
Multimodal Assistant API
========================

A production-ready FastAPI service for image analysis combining:
- Image captioning using BLIP (Bootstrapped Language-Image Pre-training)
- Optical Character Recognition (OCR) using Tesseract
- Visual Question Answering capabilities

Author: Professional Development Team
Version: 1.0.0
License: MIT
"""

import io
import logging
import time
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pytesseract

# Constants
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

# Configure professional logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multimodal Assistant API",
    description="Enterprise-grade image analysis API with captioning and OCR capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware for production deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load BLIP model for image captioning/Q&A
try:
    logger.info("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)
    logger.info("BLIP model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load BLIP model: {e}")
    raise RuntimeError(f"Model initialization failed: {e}")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url} - {response.status_code} - {process_time:.4f}s"
    )
    return response


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "message": "Multimodal Assistant API is running",
        "version": "1.0.0",
        "status": "healthy",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check with model status"""
    return {
        "status": "healthy",
        "services": {
            "blip_model": "loaded" if model is not None else "error",
            "ocr_engine": "available",
        },
        "timestamp": time.time(),
    }


def validate_image(image: UploadFile) -> None:
    """Validate uploaded image file"""
    if not image.content_type or image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    if image.size and image.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB",
        )


@app.post("/v1/vision", tags=["Vision"])
async def vision_endpoint(
    image: UploadFile = File(..., description="Image file for analysis"),
    prompt: str = Form(
        "Describe this image", description="Question or instruction for the AI"
    ),
) -> Dict[str, Any]:
    """
    Analyze an image with AI-powered captioning and OCR

    **Parameters:**
    - **image**: Image file (JPEG, PNG, WebP, BMP, TIFF)
    - **prompt**: Optional text prompt for specific questions

    **Returns:**
    - **caption**: AI-generated description or answer
    - **ocr_text**: Text extracted from the image
    - **prompt**: The input prompt used
    - **filename**: Original filename
    - **status**: Processing status
    - **processing_time**: Time taken in seconds
    """
    start_time = time.time()

    try:
        # Validate input
        validate_image(image)

        if not prompt.strip():
            prompt = "Describe this image"

        # Read and process image
        img_bytes = await image.read()
        if len(img_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # OCR processing with enhanced error handling
        ocr_text = ""
        try:
            ocr_text = pytesseract.image_to_string(img, config="--psm 6").strip()
            if not ocr_text:
                ocr_text = "No text detected in image"
        except Exception as e:
            logger.warning(f"OCR processing failed for {image.filename}: {e}")
            if "tesseract is not installed" in str(e).lower():
                ocr_text = "OCR unavailable (Tesseract not installed)"
            else:
                ocr_text = "OCR processing unavailable"

        # BLIP caption/Q&A with enhanced parameters
        try:
            inputs = processor(img, prompt, return_tensors="pt")
            output = model.generate(
                **inputs,
                max_length=100,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
            )
            caption = processor.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"BLIP processing failed for {image.filename}: {e}")
            raise HTTPException(status_code=500, detail="AI processing failed")

        processing_time = time.time() - start_time
        logger.info(
            f"Successfully processed {image.filename} in {processing_time:.2f}s"
        )

        return JSONResponse(
            {
                "caption": caption,
                "ocr_text": ocr_text,
                "prompt": prompt,
                "filename": image.filename or "unknown",
                "status": "success",
                "processing_time": round(processing_time, 3),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Unexpected error processing {image.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred during image processing",
        )


if __name__ == "__main__":
    import uvicorn  # pragma: no cover

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")  # pragma: no cover
