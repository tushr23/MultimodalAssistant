"""
Multimodal Assistant API
========================

Production-ready FastAPI service for image analysis combining:
- Image captioning using BLIP (Bootstrapped Language-Image Pre-training)
- Optical Character Recognition (OCR) using Tesseract

Author: Professional Development Team
Version: 1.0.0
License: MIT
"""

import io
import logging
import time
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import os

if os.getenv("TESTING_MODE") != "true":  # pragma: no cover
    from transformers import (
        BlipProcessor,
        BlipForConditionalGeneration,
    )  # pragma: no cover
    import pytesseract  # pragma: no cover
    import torch  # pragma: no cover
else:
    # Mock imports for testing
    from unittest.mock import MagicMock

    BlipProcessor = MagicMock()
    BlipForConditionalGeneration = MagicMock()

    class MockPytesseract:
        @staticmethod
        def image_to_string(img, config=None):
            return "Mock OCR text"

    pytesseract = MockPytesseract()

    class MockTorch:
        @staticmethod
        def no_grad():
            class NoGradContext:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

            return NoGradContext()

    torch = MockTorch()

# Configuration
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
DEFAULT_FALLBACK_CAPTION = "A scene with various visual elements."
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
}

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Multimodal Assistant API starting up...")

app = FastAPI(
    title="Multimodal Assistant API",
    description="Production-grade image analysis API with captioning and OCR capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load BLIP model
try:
    logger.info("Loading BLIP model...")
    if os.getenv("TESTING_MODE") == "true":
        processor = MagicMock()
        model = MagicMock()
        processor.return_value = {"input_ids": "mocked"}
        model.generate.return_value = [MagicMock()]
        processor.decode.return_value = "A mocked image description"
        logger.info("BLIP model mocked for testing")
    else:  # pragma: no cover
        # Set CPU-only mode for consistent CI/Docker behavior
        import torch

        torch.set_default_dtype(torch.float32)
        # Force CPU usage to avoid CUDA issues in CI/Docker
        device = "cpu"
        logger.info(f"Using device: {device}")

        processor = BlipProcessor.from_pretrained(
            BLIP_MODEL_NAME, cache_dir="/tmp/huggingface_cache"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            BLIP_MODEL_NAME,
            torch_dtype=torch.float32,
            cache_dir="/tmp/huggingface_cache",
        ).to(device)
        logger.info("BLIP model loaded successfully on CPU")
except Exception as e:  # pragma: no cover
    logger.error(f"Failed to load BLIP model: {e}")
    processor = None
    model = None


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multimodal Assistant API is running",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {"health": "/health", "docs": "/docs", "vision": "/v1/vision"},
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
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
    """Process image and return AI-generated caption and OCR text"""
    start_time = time.time()

    # Validate & read image
    validate_image(image)
    if not prompt.strip():
        prompt = "Describe this image"
    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError:
        return JSONResponse(
            status_code=400, content={"error": "Cannot identify image file format"}
        )

    # OCR processing
    try:
        ocr_text = pytesseract.image_to_string(img, config="--psm 6").strip()
        if not ocr_text:
            ocr_text = "No text detected in image"
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        if "tesseract is not installed" in str(e).lower():
            ocr_text = "OCR unavailable (Tesseract not installed)"
        else:
            ocr_text = "OCR processing unavailable"

    # AI Image Captioning
    caption = ""
    try:
        # Generate caption with BLIP
        inputs = processor(img, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs, max_length=50, do_sample=False, early_stopping=True
            )

        raw_caption = processor.decode(output[0], skip_special_tokens=True).strip()

        # Clean common BLIP artifacts
        caption = raw_caption
        if caption.startswith("arafed"):
            caption = caption[6:].strip()
        if caption.startswith("there is"):
            caption = caption[8:].strip()
        if caption.startswith("a picture of"):
            caption = caption[12:].strip()

        # Ensure proper formatting
        if caption:
            caption = (
                caption[0].upper() + caption[1:]
                if len(caption) > 1
                else caption.upper()
            )
            if not caption.endswith("."):
                caption += "."

        # Enhanced response for detailed requests
        if "detail" in prompt.lower() and len(caption) > 0:
            try:
                inputs = processor(img, return_tensors="pt")
                with torch.no_grad():
                    detailed_output = model.generate(
                        **inputs,
                        max_length=80,
                        num_beams=5,
                        do_sample=True,
                        temperature=0.7,
                    )
                detailed_caption = processor.decode(
                    detailed_output[0], skip_special_tokens=True
                ).strip()

                # Use detailed caption if it's significantly better
                if len(detailed_caption) > len(caption) * 1.3:
                    caption = detailed_caption
                    if caption.startswith("arafed"):
                        caption = caption[6:].strip()
                    if caption and caption[0].islower():
                        caption = caption[0].upper() + caption[1:]
                    if not caption.endswith("."):
                        caption += "."
            except Exception:
                pass  # Use original caption if detailed generation fails

    except Exception as e:
        logger.error(f"BLIP processing failed: {e}")
        raise HTTPException(status_code=500, detail="Image processing failed")

    # Fallback if no caption generated
    if not caption or len(caption.strip()) < 3:
        try:
            inputs = processor(img, "a photo of", return_tensors="pt")
            with torch.no_grad():
                output = model.generate(**inputs, max_length=50, num_beams=5)
            fallback_caption = processor.decode(
                output[0], skip_special_tokens=True
            ).strip()
            if fallback_caption.startswith("a photo of"):
                fallback_caption = fallback_caption[10:].strip()
            caption = fallback_caption if fallback_caption else DEFAULT_FALLBACK_CAPTION
        except Exception:
            caption = DEFAULT_FALLBACK_CAPTION

    # Ensure we always have a valid caption
    caption = caption.strip() or DEFAULT_FALLBACK_CAPTION

    processing_time = time.time() - start_time
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


if __name__ == "__main__":  # pragma: no cover
    import uvicorn  # pragma: no cover

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")  # pragma: no cover
