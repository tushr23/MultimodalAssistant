"""
Production Test Suite for Multimodal Assistant API
=================================================

Comprehensive test coverage ensuring 100% code coverage for production deployment.
"""

import os

# Set testing mode before importing main
os.environ["TESTING_MODE"] = "true"

import pytest
import io
from unittest.mock import patch, MagicMock, Mock
from fastapi.testclient import TestClient
from fastapi import HTTPException
from PIL import Image

import main

client = TestClient(main.app)


def create_test_image(width=100, height=100, color="red"):
    """Create a test image for testing"""
    img = Image.new("RGB", (width, height), color=color)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


class TestRootEndpoints:
    """Test root and health endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint returns correct information"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Multimodal Assistant API is running"
        assert data["version"] == "1.0.0"
        assert data["status"] == "healthy"
        assert "endpoints" in data

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "timestamp" in data


class TestVisionEndpoint:
    """Test vision processing endpoint"""

    def test_successful_vision_request(self):
        """Test successful image processing with default prompt"""
        test_image = create_test_image()
        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "Describe this image"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "caption" in data
        assert "ocr_text" in data
        assert data["prompt"] == "Describe this image"
        assert data["status"] == "success"
        assert "processing_time" in data

    def test_empty_prompt_handling(self):
        """Test empty prompt gets default value"""
        test_image = create_test_image()
        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": ""},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prompt"] == "Describe this image"

    def test_whitespace_prompt_handling(self):
        """Test whitespace-only prompt gets default value"""
        test_image = create_test_image()
        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "   \t\n  "},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prompt"] == "Describe this image"

    def test_detailed_prompt_processing(self):
        """Test detailed prompt triggers enhanced processing"""
        test_image = create_test_image()
        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "Describe this image in detail"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "caption" in data


class TestFileValidation:
    """Test file validation and error handling"""

    def test_invalid_content_type(self):
        """Test rejection of invalid file types"""
        text_file = io.BytesIO(b"Not an image")
        response = client.post(
            "/v1/vision",
            files={"image": ("test.txt", text_file, "text/plain")},
            data={"prompt": "test"},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_large_file_rejection(self):
        """Test large file rejection"""
        large_content = b"x" * (main.MAX_FILE_SIZE + 1000)
        large_file = io.BytesIO(large_content)
        response = client.post(
            "/v1/vision",
            files={"image": ("large.jpg", large_file, "image/jpeg")},
            data={"prompt": "test"},
        )
        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]

    def test_empty_file_handling(self):
        """Test empty file handling"""
        empty_file = io.BytesIO(b"")
        response = client.post(
            "/v1/vision",
            files={"image": ("empty.jpg", empty_file, "image/jpeg")},
            data={"prompt": "test"},
        )
        assert response.status_code == 400
        assert "Empty image file" in response.json()["detail"]

    def test_all_supported_content_types(self):
        """Test all allowed content types are accepted"""
        for content_type in main.ALLOWED_CONTENT_TYPES:
            test_image = create_test_image()
            response = client.post(
                "/v1/vision",
                files={"image": ("test_file", test_image, content_type)},
                data={"prompt": "test"},
            )
            assert response.status_code == 200

    def test_corrupted_image_handling(self):
        """Test corrupted image data handling"""
        corrupted_data = io.BytesIO(b"Not valid image data but pretends to be")
        response = client.post(
            "/v1/vision",
            files={"image": ("bad.jpg", corrupted_data, "image/jpeg")},
            data={"prompt": "test"},
        )
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data
        assert "Cannot identify image file format" in response_data["error"]


class TestOCRProcessing:
    """Test OCR functionality and error handling"""

    @patch("main.pytesseract.image_to_string")
    def test_successful_ocr(self, mock_ocr):
        """Test successful OCR processing"""
        mock_ocr.return_value = "Sample text from image"
        test_image = create_test_image()
        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ocr_text"] == "Sample text from image"

    @patch("main.pytesseract.image_to_string")
    def test_empty_ocr_result(self, mock_ocr):
        """Test empty OCR result handling"""
        mock_ocr.return_value = ""
        test_image = create_test_image()
        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ocr_text"] == "No text detected in image"

    @patch("main.pytesseract.image_to_string")
    def test_tesseract_not_installed_error(self, mock_ocr):
        """Test Tesseract not installed error handling"""
        mock_ocr.side_effect = Exception("tesseract is not installed")
        test_image = create_test_image()
        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ocr_text"] == "OCR unavailable (Tesseract not installed)"

    @patch("main.pytesseract.image_to_string")
    def test_tesseract_uppercase_error(self, mock_ocr):
        """Test Tesseract error with uppercase"""
        mock_ocr.side_effect = Exception("Tesseract is not installed")
        test_image = create_test_image()
        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ocr_text"] == "OCR unavailable (Tesseract not installed)"

    @patch("main.pytesseract.image_to_string")
    def test_generic_ocr_error(self, mock_ocr):
        """Test generic OCR error handling"""
        mock_ocr.side_effect = Exception("Some other OCR error")
        test_image = create_test_image()
        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ocr_text"] == "OCR processing unavailable"


class TestBLIPProcessing:
    """Tests for BLIP model processing scenarios"""

    @patch("main.model")
    @patch("main.processor")
    def test_blip_processing_error(self, mock_processor, mock_model):
        """Test BLIP processing failure handling"""
        mock_processor.side_effect = Exception("BLIP model error")
        test_image = create_test_image()

        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "test"},
        )
        assert response.status_code == 500
        assert "Image processing failed" in response.json()["detail"]

    @patch("main.model")
    @patch("main.processor")
    def test_detailed_prompt_processing_with_longer_caption(
        self, mock_processor, mock_model
    ):
        """Test detailed processing when detailed caption is significantly longer"""
        # Mock the processor and model
        mock_inputs = {"pixel_values": "mock_tensor"}
        mock_processor.return_value = mock_inputs

        # First call returns short caption, detailed call returns longer caption
        mock_model.generate.side_effect = [
            ["short_output"],  # First generation (regular)
            [
                "much_longer_detailed_output_that_is_significantly_more_detailed"
            ],  # Detailed generation
        ]

        # Mock processor decode calls
        mock_processor.decode.side_effect = [
            "short caption",  # Regular caption
            "much longer detailed caption that provides comprehensive description",  # Detailed caption
        ]

        test_image = create_test_image()

        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "give me detailed analysis"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "detailed" in data["caption"].lower() or len(data["caption"]) > 20

    @patch("main.model")
    @patch("main.processor")
    def test_detailed_processing_exception_fallback(self, mock_processor, mock_model):
        """Test detailed processing falls back to original caption on exception"""
        # Mock the processor and model for regular processing
        mock_inputs = {"pixel_values": "mock_tensor"}
        mock_processor.return_value = mock_inputs
        mock_model.generate.return_value = ["regular_output"]
        mock_processor.decode.return_value = "regular caption"

        # Make detailed processing fail
        def side_effect_generate(**kwargs):
            if kwargs.get("num_beams"):  # Detailed generation call
                raise Exception("Detailed generation failed")
            return ["regular_output"]

        mock_model.generate.side_effect = side_effect_generate

        test_image = create_test_image()

        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "give me detailed analysis"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["caption"] == "Regular caption."

    @patch("main.model")
    @patch("main.processor")
    def test_caption_cleaning_arafed_prefix(self, mock_processor, mock_model):
        """Test caption cleaning removes 'arafed' prefix"""
        mock_inputs = {"pixel_values": "mock_tensor"}
        mock_processor.return_value = mock_inputs
        mock_model.generate.return_value = ["output_with_arafed"]
        mock_processor.decode.return_value = "arafed a beautiful landscape"

        test_image = create_test_image()

        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["caption"] == "A beautiful landscape."

    @patch("main.model")
    @patch("main.processor")
    def test_caption_cleaning_there_is_prefix(self, mock_processor, mock_model):
        """Test caption cleaning removes 'there is' prefix"""
        mock_inputs = {"pixel_values": "mock_tensor"}
        mock_processor.return_value = mock_inputs
        mock_model.generate.return_value = ["output"]
        mock_processor.decode.return_value = "there is a beautiful sunset"

        test_image = create_test_image()

        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["caption"] == "A beautiful sunset."

    @patch("main.model")
    @patch("main.processor")
    def test_caption_cleaning_a_picture_of_prefix(self, mock_processor, mock_model):
        """Test caption cleaning removes 'a picture of' prefix"""
        mock_inputs = {"pixel_values": "mock_tensor"}
        mock_processor.return_value = mock_inputs
        mock_model.generate.return_value = ["output"]
        mock_processor.decode.return_value = "a picture of a beautiful garden"

        test_image = create_test_image()

        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["caption"] == "A beautiful garden."

    @patch("main.model")
    @patch("main.processor")
    def test_fallback_caption_generation(self, mock_processor, mock_model):
        """Test fallback caption generation when main caption is too short"""
        # Mock the processor and model for main processing (returns short caption)
        mock_inputs = {"pixel_values": "mock_tensor"}
        mock_processor.return_value = mock_inputs

        # First call returns very short/empty caption, second call (fallback) returns proper caption
        mock_model.generate.side_effect = [
            ["short_output"],  # Main generation (too short)
            ["fallback_output"],  # Fallback generation
        ]

        mock_processor.decode.side_effect = [
            "x",  # Too short caption (< 3 chars)
            "a photo of beautiful scenery",  # Fallback caption
        ]

        test_image = create_test_image()

        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["caption"] == "beautiful scenery"

    @patch("main.model")
    @patch("main.processor")
    def test_fallback_caption_exception_handling(self, mock_processor, mock_model):
        """Test fallback caption when both main and fallback generation fail"""
        # Mock the processor and model for main processing
        mock_inputs = {"pixel_values": "mock_tensor"}
        mock_processor.return_value = mock_inputs

        # First call returns empty caption, second call (fallback) throws exception
        mock_model.generate.side_effect = [
            ["short_output"],  # Main generation (empty)
            Exception("Fallback generation failed"),  # Fallback generation fails
        ]

        mock_processor.decode.return_value = ""  # Empty caption

        test_image = create_test_image()

        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["caption"] == "A scene with various visual elements."

    @patch("main.model")
    @patch("main.processor")
    def test_single_character_caption_formatting(self, mock_processor, mock_model):
        """Test single character caption gets properly formatted"""
        mock_inputs = {"pixel_values": "mock_tensor"}
        mock_processor.return_value = mock_inputs
        mock_model.generate.return_value = ["single_char_output"]
        mock_processor.decode.return_value = "a"

        test_image = create_test_image()

        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["caption"] == "a"


class TestValidationFunction:
    """Test the validate_image function directly"""

    def test_validate_image_invalid_type(self):
        """Test validate_image function with invalid content type"""
        mock_file = MagicMock()
        mock_file.content_type = "text/plain"
        mock_file.size = 1000

        with pytest.raises(Exception):
            main.validate_image(mock_file)

    def test_validate_image_oversized_file(self):
        """Test validate_image function with oversized file"""
        mock_file = MagicMock()
        mock_file.content_type = "image/jpeg"
        mock_file.size = main.MAX_FILE_SIZE + 1000

        with pytest.raises(Exception):
            main.validate_image(mock_file)


class TestApplicationConfiguration:
    """Tests for application configuration and setup"""

    def test_app_configuration(self):
        """Test FastAPI application is properly configured"""
        from main import app

        assert app.title == "Multimodal Assistant API"
        assert hasattr(app, "routes")
        assert len(app.routes) > 0

    def test_configuration_constants(self):
        """Test configuration constants are properly defined"""
        from main import MAX_FILE_SIZE, ALLOWED_CONTENT_TYPES

        assert MAX_FILE_SIZE == 10 * 1024 * 1024  # 10MB
        assert isinstance(ALLOWED_CONTENT_TYPES, set)
        assert "image/jpeg" in ALLOWED_CONTENT_TYPES


class TestExtendedCoverage:
    """Additional tests to achieve 100% coverage"""

    @patch("main.model", None)
    def test_health_check_with_model_error(self):
        """Test health check when model is None"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["services"]["blip_model"] == "error"

    def test_validate_image_with_none_content_type(self):
        """Test validate_image with None content_type"""
        from main import validate_image
        import tempfile

        # Create a mock file with None content_type
        mock_file = Mock()
        mock_file.content_type = None
        mock_file.filename = "test.jpg"

        with pytest.raises(HTTPException) as exc_info:
            validate_image(mock_file)
        assert exc_info.value.status_code == 400
        assert "Unsupported file type" in exc_info.value.detail

    @patch("main.model")
    @patch("main.processor")
    def test_vision_empty_caption_fallback_scenarios(self, mock_processor, mock_model):
        """Test various empty caption fallback scenarios"""
        # Test case 1: Empty caption from main processing, successful fallback
        mock_inputs = {"pixel_values": "mock_tensor"}
        mock_processor.return_value = mock_inputs

        mock_model.generate.side_effect = [
            ["main_output"],  # Main generation (empty result)
            ["fallback_output"],  # Fallback generation
        ]

        mock_processor.decode.side_effect = [
            "",  # Empty main caption
            "a photo of nice view",  # Fallback caption
        ]

        test_image = create_test_image()

        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["caption"] == "nice view"

    @patch("main.model")
    @patch("main.processor")
    def test_vision_final_fallback_default(self, mock_processor, mock_model):
        """Test final fallback to default caption"""
        mock_inputs = {"pixel_values": "mock_tensor"}
        mock_processor.return_value = mock_inputs

        # Both main and fallback generation return empty
        mock_model.generate.side_effect = [
            ["main_output"],  # Main generation
            ["fallback_output"],  # Fallback generation
        ]

        mock_processor.decode.side_effect = [
            "",  # Empty main caption
            "",  # Empty fallback caption too
        ]

        test_image = create_test_image()

        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["caption"] == "A scene with various visual elements."

    @patch("main.model")
    @patch("main.processor")
    def test_detailed_caption_arafed_cleaning_and_capitalization(
        self, mock_processor, mock_model
    ):
        """Test detailed caption processing with arafed cleaning and capitalization"""
        mock_inputs = {"pixel_values": "mock_tensor"}
        mock_processor.return_value = mock_inputs

        # First call returns short caption, detailed call returns longer caption with issues
        mock_model.generate.side_effect = [
            ["short_output"],  # First generation (regular)
            ["detailed_output"],  # Detailed generation (much longer)
        ]

        # Mock processor decode calls - detailed caption has arafed prefix and starts lowercase
        mock_processor.decode.side_effect = [
            "short",  # Regular caption
            "arafed beautiful landscape with mountains and trees",  # Detailed caption with issues
        ]

        test_image = create_test_image()

        response = client.post(
            "/v1/vision",
            files={"image": ("test.jpg", test_image, "image/jpeg")},
            data={"prompt": "give me detailed description"},
        )

        assert response.status_code == 200
        data = response.json()
        # Should have cleaned "arafed" prefix and capitalized first letter
        assert data["caption"] == "Beautiful landscape with mountains and trees."


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main(
        [
            __file__,
            "--cov=main",
            "--cov-report=term-missing",
            "--cov-fail-under=100",
            "-v",
        ]
    )
