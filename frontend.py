#!/usr/bin/env python3
"""
Multimodal Assistant Frontend Client
====================================

Interactive command-line interface for the Multimodal Assistant API.
Provides a user-friendly way to upload images and get AI-powered analysis.

Features:
- Image upload and validation
- Interactive prompting
- Formatted result display
- Error handling and retry logic
- Environment-aware configuration

Author: Professional Development Team
Version: 1.0.0
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000/v1/vision")
HEALTH_URL = os.environ.get("API_URL", "http://localhost:8000").replace("/v1/vision", "/health")
MAX_RETRIES = 3
TIMEOUT = 30
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif'}

class MultimodalClient:
    """Professional client for Multimodal Assistant API"""
    
    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy"""
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
        
    def check_health(self) -> bool:
        """Check if the API is healthy and responsive"""
        try:
            response = self.session.get(HEALTH_URL, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
            
    def analyze_image(self, image_path: str, prompt: str = "Describe this image") -> Dict[str, Any]:
        """Send image to API for analysis"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Validate file format
        file_ext = Path(image_path).suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format {file_ext}. Supported: {', '.join(SUPPORTED_FORMATS)}")
            
        try:
            with open(image_path, "rb") as img_file:
                files = {"image": (os.path.basename(image_path), img_file, "image/jpeg")}
                data = {"prompt": prompt}
                
                response = self.session.post(
                    self.base_url, 
                    files=files, 
                    data=data, 
                    timeout=TIMEOUT
                )
                response.raise_for_status()
                return response.json()
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")

def print_banner():
    """Display application banner"""
    print("="*60)
    print("🤖 Multimodal Assistant - AI Image Analysis")
    print("   Professional Image Captioning & OCR Service")
    print("="*60)

def print_result(result: Dict[str, Any]):
    """Format and display analysis results"""
    print("\n" + "="*50)
    print("📋 ANALYSIS RESULTS")
    print("="*50)
    
    print(f"📁 File: {result.get('filename', 'Unknown')}")
    print(f"❓ Prompt: {result.get('prompt', 'N/A')}")
    print(f"⏱️  Processing Time: {result.get('processing_time', 'N/A')}s")
    
    print(f"\n🎨 AI Caption:")
    print(f"   {result.get('caption', 'No caption generated')}")
    
    ocr_text = result.get('ocr_text', '').strip()
    if ocr_text and ocr_text != "No text detected in image":
        print(f"\n📝 Extracted Text:")
        print(f"   {ocr_text}")
    else:
        print(f"\n📝 Extracted Text: No text detected")
    
    print("="*50)

def get_user_input() -> tuple[str, str]:
    """Get image path and prompt from user with validation"""
    while True:
        print("\n📸 Enter image details:")
        image_path = input("Image file path: ").strip().strip('"\'')
        
        if not image_path:
            print("❌ Please enter a valid file path")
            continue
            
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            continue
            
        file_ext = Path(image_path).suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            print(f"❌ Unsupported format. Use: {', '.join(SUPPORTED_FORMATS)}")
            continue
            
        break
    
    prompt = input("Your question/prompt (Enter for default): ").strip()
    if not prompt:
        prompt = "Describe this image"
        
    return image_path, prompt

def main():
    """Main application loop"""
    print_banner()
    
    # Initialize client
    client = MultimodalClient()
    
    # Health check
    print("🔍 Checking API health...")
    if not client.check_health():
        print("❌ API is not responding. Please ensure the backend is running.")
        print(f"   Expected URL: {HEALTH_URL}")
        sys.exit(1)
    print("✅ API is healthy and ready")
    
    try:
        while True:
            try:
                # Get user input
                image_path, prompt = get_user_input()
                
                # Process image
                print(f"\n🚀 Analyzing image with AI...")
                start_time = time.time()
                
                result = client.analyze_image(image_path, prompt)
                
                # Display results
                print_result(result)
                
                # Ask for another analysis
                print(f"\n🔄 Analyze another image? (y/N): ", end="")
                if input().lower() not in ['y', 'yes']:
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("🔄 Please try again or check your input.\n")
                
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    
    print("\n✨ Thank you for using Multimodal Assistant!")

if __name__ == "__main__":
    main()
