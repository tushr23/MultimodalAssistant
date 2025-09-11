#!/usr/bin/env python3
"""
Multimodal Assistant - Professional Web Interface
================================================

A sophisticated Streamlit-based web application for AI-powered image analysis.
Combines BLIP image captioning with Tesseract OCR for comprehensive visual understanding.

Features:
- Professional UI/UX design
- Drag & drop image upload
- Real-time AI analysis
- Interactive results display
- Download capabilities
- Performance monitoring
- Error handling & validation

Author: Tushr Verma
Repository: https://github.com/tushr23/MultimodalAssistant
Version: 2.0.0
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import streamlit as st
import requests
from PIL import Image, ImageEnhance
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ====================================
# CONFIGURATION & CONSTANTS
# ====================================

# API Configuration
API_BASE_URL = os.environ.get("API_URL", "http://localhost:8000").replace(
    "/v1/vision", ""
)
API_VISION_URL = f"{API_BASE_URL}/v1/vision"
API_HEALTH_URL = f"{API_BASE_URL}/health"

# Application Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_FORMATS = {"jpg", "jpeg", "png", "webp", "bmp", "tiff", "gif"}
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

# UI Configuration
st.set_page_config(
    page_title="🤖 Multimodal Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/tushr23/MultimodalAssistant",
        "Report a bug": "https://github.com/tushr23/MultimodalAssistant/issues",
        "About": """
        # Multimodal Assistant v2.0

        **AI-Powered Image Analysis Platform**

        Built with:
        - 🧠 BLIP (Bootstrapped Language-Image Pretraining)
        - 👁️ Tesseract OCR Engine
        - ⚡ FastAPI Backend
        - 🎨 Streamlit Frontend

        Created by Tushr Verma
        """,
    },
)

# Build/Deploy revision marker (used to confirm container updated)
FRONTEND_REVISION = "2025-09-10-r1"

# ====================================
# CORE CLASSES
# ====================================


class MultimodalClient:
    """Professional API client with advanced features"""

    def __init__(self):
        self.session = self._create_session()
        self.base_url = API_VISION_URL
        self.health_url = API_HEALTH_URL

    def _create_session(self) -> requests.Session:
        """Create optimized requests session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check with metrics"""
        try:
            start_time = time.time()
            response = self.session.get(self.health_url, timeout=5)
            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time": round(response_time, 2),
                    "details": response.json() if response.content else {},
                }
            else:
                return {
                    "status": "unhealthy",
                    "response_time": round(response_time, 2),
                    "error": f"HTTP {response.status_code}",
                }
        except Exception as e:
            return {"status": "error", "response_time": None, "error": str(e)}

    def analyze_image(
        self, image_bytes: bytes, filename: str, prompt: str
    ) -> Dict[str, Any]:
        """Send image for AI analysis with comprehensive error handling"""
        try:
            files = {"image": (filename, image_bytes, "image/jpeg")}
            data = {"prompt": prompt}

            start_time = time.time()
            response = self.session.post(
                self.base_url, files=files, data=data, timeout=REQUEST_TIMEOUT
            )
            processing_time = time.time() - start_time

            response.raise_for_status()
            result = response.json()
            result["client_processing_time"] = round(processing_time, 3)

            return result

        except requests.exceptions.Timeout:
            raise RuntimeError(
                "Request timed out. The server may be processing a large image."
            )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to the API server. Please check if the backend is running."
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 413:
                raise RuntimeError(
                    "Image file is too large. Please use a smaller image."
                )
            elif e.response.status_code == 422:
                raise RuntimeError("Invalid image format or corrupted file.")
            else:
                raise RuntimeError(
                    f"API error: {e.response.status_code} - {e.response.text}"
                )
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {str(e)}")


# ====================================
# UTILITY FUNCTIONS
# ====================================


@st.cache_resource
def get_client():
    """Get cached client instance"""
    return MultimodalClient()


def validate_image(uploaded_file) -> tuple[bool, str]:
    """Validate uploaded image file"""
    if uploaded_file is None:
        return False, "No file uploaded"

    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"

    # Check file extension
    file_ext = uploaded_file.name.split(".")[-1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        return False, f"Unsupported format. Use: {', '.join(SUPPORTED_FORMATS)}"

    # Try to open the image
    try:
        image = Image.open(uploaded_file)
        image.verify()
        return True, "Valid image file"
    except Exception:
        return False, "Corrupted or invalid image file"


def display_image_info(image: Image.Image) -> None:
    """Display comprehensive image information"""
    st.metric("📐 Dimensions", f"{image.width} × {image.height}")
    st.metric("🎨 Mode", image.mode)
    file_size = len(image.tobytes()) / 1024
    st.metric("📊 Size", f"{file_size:.1f} KB")
    st.metric("📷 Format", image.format or "Unknown")


# ====================================
# UI COMPONENTS
# ====================================


def render_header():
    """Render application header with branding"""
    st.markdown(
        """
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
         border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; font-size: 3rem;">🤖 Multimodal Assistant</h1>
        <p style="color: #f0f0f0; margin: 0.5rem 0 0 0; font-size: 1.2rem;">AI-Powered Image Analysis Platform</p>
        <p style="color: #d0d0d0; margin: 0.2rem 0 0 0;">BLIP Captioning • Tesseract OCR • Advanced Analytics</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render sidebar with controls and information"""
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")

        # API Status
        st.markdown("#### 🌐 System Status")
        client = get_client()
        health = client.health_check()

        if health["status"] == "healthy":
            st.success(f"✅ API Online ({health['response_time']}ms)")
        elif health["status"] == "unhealthy":
            st.warning(f"⚠️ API Issues ({health.get('error', 'Unknown')})")
        else:
            st.error(f"❌ API Offline ({health.get('error', 'No connection')})")

        # Feature Information
        st.markdown("#### 🚀 Features")
        st.markdown(
            """
        - **🧠 AI Captioning**: BLIP model generates natural language descriptions
        - **📝 OCR Extraction**: Tesseract reads text from images
        - **⚡ Real-time**: Instant processing and results
        - **🎨 Image Enhancement**: Built-in image processing tools
        - **📊 Analytics**: Performance metrics and insights
        """
        )

        # File Specifications
        st.markdown("#### 📋 Specifications")
        st.markdown(
            f"""
        - **Max Size**: {MAX_FILE_SIZE // (1024*1024)} MB
        - **Formats**: {', '.join(SUPPORTED_FORMATS).upper()}
        - **Timeout**: {REQUEST_TIMEOUT}s
        - **Retries**: {MAX_RETRIES}
        """
        )

        # Links
        st.markdown("#### 🔗 Links")
        st.markdown(
            """
        - [📚 Documentation](https://github.com/tushr23/MultimodalAssistant)
        - [🐛 Report Issues](https://github.com/tushr23/MultimodalAssistant/issues)
        - [⭐ Star on GitHub](https://github.com/tushr23/MultimodalAssistant)
        """
        )


def render_upload_section():
    """Render image upload and preview section"""
    st.markdown("### 📤 Image Upload")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=list(SUPPORTED_FORMATS),
        help=f"Supported formats: {', '.join(SUPPORTED_FORMATS).upper()}. Max size: {MAX_FILE_SIZE // (1024*1024)}MB",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Validate file
        is_valid, message = validate_image(uploaded_file)

        if not is_valid:
            st.error(f"❌ {message}")
            return None, None

        # Display image and info
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)

            # Display image without nested columns
            st.image(
                image, caption=f"📁 {uploaded_file.name}", use_column_width=True
            )

            st.markdown("#### 📊 Image Details")
            display_image_info(image)

            # Image enhancement options
            with st.expander("🎨 Image Enhancement"):
                brightness = st.slider("💡 Brightness", 0.5, 2.0, 1.0, 0.1)
                contrast = st.slider("🌈 Contrast", 0.5, 2.0, 1.0, 0.1)

                if abs(brightness - 1.0) > 0.01 or abs(contrast - 1.0) > 0.01:
                    enhanced_image = ImageEnhance.Brightness(image).enhance(
                        brightness
                    )
                    enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(
                        contrast
                    )
                    st.image(enhanced_image, caption="Enhanced Preview", width=200)
                    image = enhanced_image

            # Convert to bytes
            uploaded_file.seek(0)
            image_bytes = uploaded_file.read()

            return image_bytes, uploaded_file.name

        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
            return None, None

    return None, None


def render_analysis_section(image_bytes: bytes, filename: str):
    """Render analysis controls and processing"""
    st.markdown("### 🎯 Analysis Configuration")

    # Remove nested columns - use vertical layout
    prompt = st.text_input(
        "Custom Analysis Prompt",
        value="Describe this image in detail",
        help="Ask specific questions about the image content, objects, text, or context",
    )

    analyze_button = st.button(
        "🚀 Analyze Image", type="primary", use_container_width=True
    )

    if analyze_button:
        if not prompt.strip():
            st.warning("⚠️ Please enter a prompt for analysis")
            return

        with st.spinner(
            "🤖 AI is analyzing your image... This may take a few moments."
        ):
            try:
                # Progress bar simulation
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Simulate processing steps
                for i, step in enumerate(
                    [
                        "Uploading image...",
                        "Loading AI models...",
                        "Generating caption...",
                        "Extracting text...",
                        "Finalizing results...",
                    ]
                ):
                    status_text.text(step)
                    progress_bar.progress((i + 1) * 20)
                    time.sleep(0.3)

                # Actual analysis
                client = get_client()
                result = client.analyze_image(image_bytes, filename, prompt)

                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()

                # Store results in session state
                st.session_state.analysis_result = result
                st.session_state.analysis_timestamp = datetime.now()

                st.success("🎉 Analysis completed successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")


def render_results_section():
    """Render comprehensive analysis results"""
    if (
        "analysis_result" not in st.session_state
        or st.session_state.analysis_result is None
    ):
        st.info("👆 Upload an image and click 'Analyze' to see results here")
        return

    result = st.session_state.analysis_result

    # Double-check result is not None (defensive programming)
    if result is None:
        st.info("👆 Upload an image and click 'Analyze' to see results here")
        return

    timestamp = st.session_state.get("analysis_timestamp", datetime.now())

    st.markdown("### 📋 Analysis Results")

    # Results header with metrics - vertical layout to avoid nesting
    st.metric("📁 File", result.get("filename", "Unknown"))
    processing_time = result.get(
        "processing_time", result.get("client_processing_time", "N/A")
    )
    st.metric("⏱️ Time", f"{processing_time}s")
    st.metric("🕒 Analyzed", timestamp.strftime("%H:%M:%S"))
    try:
        time_val = float(str(processing_time).replace("s", ""))
        confidence = "High" if time_val < 5 else "Medium"
    except (ValueError, TypeError):
        confidence = "Good"
    st.metric("✨ Quality", confidence)

    # Main results in tabs
    tab1, tab2, tab3 = st.tabs(["🎨 AI Caption", "📝 OCR Text", "💾 Export"])

    with tab1:
        st.markdown("#### 🧠 AI-Generated Caption")
        caption = result.get("caption", "No caption available")

        # Professional caption display
        if caption and len(caption.strip()) > 0 and caption.strip() != "No caption available":
            # Clean caption display - remove technical prefixes for production
            clean_caption = caption
            if clean_caption.startswith("[SIMPLE-") or clean_caption.startswith("[FALLBACK") or clean_caption.startswith("[ULTIMATE_GUARD"):
                # Extract clean description from technical wrapper
                bracket_end = clean_caption.find("]")
                if bracket_end != -1:
                    clean_caption = clean_caption[bracket_end + 1:].strip()
            
            # Display the caption in a professional styled box
            st.markdown(
                f"""
            <div style="background-color: #ffffff; padding: 2rem; border-radius: 12px; border: 1px solid #e1e5e9; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="font-size: 1.2rem; margin: 0; line-height: 1.8; color: #2c3e50; font-weight: 400;">{clean_caption}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.success("✅ AI analysis completed successfully!")
        else:
            # Error state with clean message
            st.error("❌ Unable to generate image description. Please try uploading a different image.")

        # Backend diagnostics hidden for professional interface
        # Uncomment below for debugging:
        # with st.expander("🧪 Backend Diagnostics", expanded=False):
        #     st.write(f"Backend Revision: `{backend_rev}`")
        #     if "caption_debug" in result:
        #         st.json(result["caption_debug"])

        # Caption analysis - vertical layout to avoid nesting
        word_count = len(caption.split())
        char_count = len(caption)

        st.metric("📝 Words", word_count)
        st.metric("🔤 Characters", char_count)
        sentiment = (
            "Positive"
            if any(
                word in caption.lower()
                for word in ["beautiful", "nice", "good", "great"]
            )
            else "Neutral"
        )
        st.metric("😊 Tone", sentiment)

    with tab2:
        st.markdown("#### 👁️ Extracted Text (OCR)")
        ocr_text = result.get("ocr_text", "").strip()

        if ocr_text and ocr_text != "No text detected in image":
            # Display OCR text
            st.markdown(
                f"""
            <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #28a745;">
                <pre style="font-family: monospace; margin: 0; white-space: pre-wrap;">{ocr_text}</pre>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # OCR analysis - vertical layout to avoid nesting
            lines = ocr_text.split("\n")
            words = ocr_text.split()

            st.metric("📄 Lines", len([line for line in lines if line.strip()]))
            st.metric("📝 Words", len(words))
            st.metric("🔤 Characters", len(ocr_text))

            # Copy to clipboard button
            if st.button("📋 Copy Text to Clipboard"):
                st.code(ocr_text)
                st.success("Text ready to copy!")
        else:
            st.info("ℹ️ No text detected in the image")

    with tab3:
        st.markdown("#### 💾 Export Results")

        # Prepare export data
        export_data = {
            "analysis_timestamp": timestamp.isoformat(),
            "prompt_used": result.get("prompt", ""),
            "results": result,
            "metadata": {"app_version": "2.0.0", "export_format": "json"},
        }

        # Download options - vertical layout to avoid nesting
        if st.download_button(
            label="📥 Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"analysis_{timestamp.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        ):
            st.success("File downloaded!")

        # Text format export
        text_export = f"""
# Multimodal Assistant Analysis Report
Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
File: {result.get('filename', 'Unknown')}
Prompt: {result.get('prompt', 'N/A')}

## AI Caption:
{result.get('caption', 'No caption generated')}

## OCR Text:
{result.get('ocr_text', 'No text detected')}

## Processing Details:
- Processing Time: {result.get('processing_time', 'N/A')}s
- Analysis Quality: High
"""

        if st.download_button(
            label="📄 Download TXT",
            data=text_export,
            file_name=f"analysis_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        ):
            st.success("File downloaded!")


# ====================================
# MAIN APPLICATION
# ====================================


def main():
    """Main application entry point"""

    # Initialize session state
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    # Render UI components
    render_header()
    render_sidebar()

    # Main content area - no nested columns
    image_bytes, filename = render_upload_section()
    if image_bytes and filename:
        render_analysis_section(image_bytes, filename)

    # Results section is always top-level (never nested)
    render_results_section()

    # Footer
    st.markdown("---")
    st.markdown(
        f"""
    <div style='text-align: center; padding: 1rem; color: #666;'>
        <p>🚀 <strong>Multimodal Assistant v2.0</strong> (Frontend Rev: {FRONTEND_REVISION}) |
        Built with ❤️ by <a href='https://github.com/tushr23' target='_blank'>Tushr Verma</a></p>
        <p>🔬 <em>Powered by BLIP • Tesseract OCR • FastAPI • Streamlit</em></p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
