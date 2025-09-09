"""
Simple test to verify basic functionality and achieve 100% coverage
"""

import pytest
import io
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image

# Mock the transformers imports to avoid model download
with patch("transformers.BlipProcessor.from_pretrained") as mock_processor, patch(
    "transformers.BlipForConditionalGeneration.from_pretrained"
) as mock_model:

    # Set up mocks
    mock_processor_instance = MagicMock()
    mock_model_instance = MagicMock()

    mock_processor.return_value = mock_processor_instance
    mock_model.return_value = mock_model_instance

    # Configure mock responses
    mock_processor_instance.return_value = {"input_ids": "mocked"}
    mock_model_instance.generate.return_value = ["mocked_output"]
    mock_processor_instance.decode.return_value = "A test image description"

    # Now import main after mocking
    import main

client = TestClient(main.app)


def create_test_image(width=100, height=100):
    """Create a test image for testing"""
    img = Image.new("RGB", (width, height), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


def test_root_endpoint():
    """Test root endpoint returns welcome message"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "status" in data


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "services" in data
    assert "timestamp" in data


def test_successful_vision_request():
    """Test successful image processing"""
    test_image = create_test_image()
    response = client.post(
        "/v1/vision", files={"image": ("test.jpg", test_image, "image/jpeg")}, data={"prompt": "Describe this image"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "caption" in data
    assert "ocr_text" in data
    assert "prompt" in data
    assert "filename" in data
    assert "status" in data
    assert "processing_time" in data


def test_invalid_content_type():
    """Test rejection of invalid file types"""
    text_file = io.BytesIO(b"Not an image")
    response = client.post("/v1/vision", files={"image": ("test.txt", text_file, "text/plain")}, data={"prompt": "test"})
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_large_file_rejection():
    """Test rejection of files larger than MAX_FILE_SIZE"""
    large_content = b"x" * (main.MAX_FILE_SIZE + 1000)
    large_file = io.BytesIO(large_content)
    response = client.post("/v1/vision", files={"image": ("large.jpg", large_file, "image/jpeg")}, data={"prompt": "test"})
    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]


def test_empty_file():
    """Test handling of empty files"""
    empty_file = io.BytesIO(b"")
    response = client.post("/v1/vision", files={"image": ("empty.jpg", empty_file, "image/jpeg")}, data={"prompt": "test"})
    assert response.status_code == 400


@patch("main.pytesseract.image_to_string")
def test_ocr_exception(mock_ocr):
    """Test OCR processing exceptions"""
    mock_ocr.side_effect = Exception("OCR failed")
    test_image = create_test_image()
    response = client.post("/v1/vision", files={"image": ("test.jpg", test_image, "image/jpeg")}, data={"prompt": "test"})
    assert response.status_code == 200
    data = response.json()
    assert "OCR processing unavailable" in data["ocr_text"] or "OCR" in data["ocr_text"]


@patch("main.pytesseract.image_to_string")
def test_tesseract_not_installed(mock_ocr):
    """Test Tesseract not installed error"""
    mock_ocr.side_effect = Exception("tesseract is not installed")
    test_image = create_test_image()
    response = client.post("/v1/vision", files={"image": ("test.jpg", test_image, "image/jpeg")}, data={"prompt": "test"})
    assert response.status_code == 200
    data = response.json()
    assert "OCR unavailable (Tesseract not installed)" in data["ocr_text"]


@patch("main.pytesseract.image_to_string")
def test_empty_ocr_result(mock_ocr):
    """Test empty OCR result handling"""
    mock_ocr.return_value = ""
    test_image = create_test_image()
    response = client.post("/v1/vision", files={"image": ("test.jpg", test_image, "image/jpeg")}, data={"prompt": "test"})
    assert response.status_code == 200
    data = response.json()
    assert data["ocr_text"] == "No text detected in image"


def test_validate_image_function():
    """Test the validate_image function directly"""
    from fastapi import UploadFile

    # Test with invalid content type
    mock_file = MagicMock(spec=UploadFile)
    mock_file.content_type = "text/plain"
    mock_file.size = 1000

    with pytest.raises(Exception):
        main.validate_image(mock_file)

    # Test with large file
    mock_file.content_type = "image/jpeg"
    mock_file.size = main.MAX_FILE_SIZE + 1

    with pytest.raises(Exception):
        main.validate_image(mock_file)

    # Test with None content type
    mock_file.content_type = None
    mock_file.size = 1000

    with pytest.raises(Exception):
        main.validate_image(mock_file)


def test_app_metadata():
    """Test app configuration"""
    assert main.app.title == "Multimodal Assistant API"
    assert main.app.version == "1.0.0"
    assert main.app.docs_url == "/docs"
    assert main.app.redoc_url == "/redoc"


def test_constants_accessibility():
    """Test that all constants are properly defined"""
    assert main.BLIP_MODEL_NAME == "Salesforce/blip-image-captioning-base"
    assert main.MAX_FILE_SIZE == 10 * 1024 * 1024
    assert len(main.ALLOWED_CONTENT_TYPES) == 5
    assert main.model is not None
    assert main.processor is not None
    assert main.logger is not None


def test_empty_prompt_default():
    """Test empty prompt gets default value"""
    test_image = create_test_image()
    response = client.post("/v1/vision", files={"image": ("test.jpg", test_image, "image/jpeg")}, data={"prompt": ""})
    assert response.status_code == 200
    data = response.json()
    assert data["prompt"] == "Describe this image"


def test_whitespace_prompt_default():
    """Test whitespace-only prompt gets default value"""
    test_image = create_test_image()
    response = client.post("/v1/vision", files={"image": ("test.jpg", test_image, "image/jpeg")}, data={"prompt": "   \n\t  "})
    assert response.status_code == 200
    data = response.json()
    assert data["prompt"] == "Describe this image"


if __name__ == "__main__":
    pytest.main([__file__, "--cov=main", "--cov-report=term-missing", "--cov-fail-under=100", "-v"])
