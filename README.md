# 🤖 Multimodal Assistant

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://hub.docker.com)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-orange.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Coverage](https://img.shields.io/badge/Coverage-100%25-brightgreen.svg)](https://github.com/tushr23/MultimodalAssistant)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Professional AI-powered image analysis platform** with BLIP captioning, Tesseract OCR, and beautiful Streamlit web interface.

## 🚀 Quick Start

**Docker Compose (Recommended)**
```bash
git clone https://github.com/tushr23/MultimodalAssistant.git
cd MultimodalAssistant
docker compose up --build
```

✅ **Backend API**: http://localhost:8000  
✅ **Web Interface**: http://localhost:8501  
✅ **API Documentation**: http://localhost:8000/docs  

## 🌟 Features

### 🎨 **Professional Web Interface**
- **Drag & Drop Upload**: Intuitive file upload with validation
- **Real-time Analysis**: Instant AI processing with progress indicators
- **Interactive Results**: Tabbed interface for captions, OCR, and analytics
- **Export Options**: Download results in JSON or TXT format
- **Image Enhancement**: Built-in brightness and contrast controls
- **Responsive Design**: Works on desktop, tablet, and mobile

### 🧠 **AI Capabilities**
- **BLIP Captioning**: Advanced image understanding and description
- **Tesseract OCR**: Professional-grade text extraction
- **Visual Q&A**: Ask specific questions about image content
- **Multi-format Support**: JPEG, PNG, WebP, BMP, TIFF, GIF
- **Performance Analytics**: Processing time and quality metrics

### 🔧 **Production Features**
- **Containerized**: Docker Compose orchestration
- **Health Monitoring**: Real-time API status and performance
- **Error Handling**: Comprehensive validation and user feedback
- **Security**: File type verification, size limits, input sanitization
- **CI/CD Ready**: GitHub Actions workflow included

## 📱 Screenshots

### Main Interface
![Main Interface](https://via.placeholder.com/800x400/667eea/ffffff?text=Professional+Web+Interface)

### Analysis Results
![Results](https://via.placeholder.com/800x400/764ba2/ffffff?text=Comprehensive+Analysis+Results)

## 🏃‍♂️ Usage

### Web Interface
1. **Access**: Open http://localhost:8501 in your browser
2. **Upload**: Drag and drop an image or click to browse
3. **Configure**: Enter a custom prompt or use the default
4. **Analyze**: Click "Analyze Image" and watch the real-time progress
5. **Review**: Explore results in the tabbed interface
6. **Export**: Download results in your preferred format

### API Usage
```bash
curl -X POST "http://localhost:8000/v1/vision" \
  -F "image=@photo.jpg" \
  -F "prompt=What's in this image?"
```

**Response:**
```json
{
  "caption": "A person walking with a dog in a sunny park",
  "ocr_text": "WELCOME TO CENTRAL PARK",
  "prompt": "What's in this image?",
  "filename": "photo.jpg",
  "processing_time": 1.2,
  "status": "success"
}
```

## ⚙️ Local Development

**Prerequisites:**
- Docker Desktop or Python 3.10+
- 4GB+ RAM (for AI models)
- 1GB+ disk space

**Manual Setup:**
```bash
# Backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend (new terminal)
streamlit run frontend.py
```

## 🧪 Testing

```bash
# Run comprehensive test suite
python test_api.py

# Run with coverage analysis
pytest test_api.py -v --cov=main --cov-report=html

# Test specific components
python test_final_coverage.py
python test_exception_paths.py
```

**Test Features:**
- ✅ 100% line coverage target
- ✅ Exception path validation
- ✅ Performance benchmarking
- ✅ Security testing
- ✅ Multi-format image support
- ✅ OCR functionality validation

## 🎯 Architecture

```mermaid
graph TB
    A[Streamlit Frontend] --> B[FastAPI Backend]
    B --> C[BLIP Model]
    B --> D[Tesseract OCR]
    E[Docker Compose] --> A
    E --> B
    F[GitHub Actions] --> G[Testing & Deployment]
```

**Components:**
- **Frontend**: Streamlit web application with professional UI
- **Backend**: FastAPI service with AI model integration
- **AI Models**: BLIP for captioning, Tesseract for OCR
- **Infrastructure**: Docker containers with health checks
- **CI/CD**: Automated testing and deployment pipeline

## 🌐 Deployment

### Cloud Platforms
- **Heroku**: `git push heroku main`
- **Railway**: Connect GitHub repository
- **Google Cloud**: `gcloud run deploy --source .`
- **AWS/Azure**: Use Docker Compose setup

### Environment Variables
```bash
API_URL=http://localhost:8000  # Backend URL for frontend
LOG_LEVEL=INFO                 # Logging level
PYTHONUNBUFFERED=1            # Python output buffering
```

## 📊 Performance

- **Model Loading**: ~30 seconds (first run only)
- **Image Processing**: 1-3 seconds average
- **Memory Usage**: ~2GB for AI models
- **Supported File Size**: Up to 10MB per image
- **Concurrent Users**: 10+ simultaneous requests

## 🔒 Security

- **Input Validation**: File type and size verification
- **Sanitization**: Filename and prompt cleaning
- **Rate Limiting**: Built-in request throttling
- **Error Handling**: Secure error messages
- **Container Security**: Non-root user execution

## 🎓 Educational Value

This project demonstrates:
- **Full-Stack Development**: Frontend, backend, and database integration
- **AI/ML Integration**: Computer vision and NLP model deployment
- **DevOps Practices**: Containerization, orchestration, and CI/CD
- **API Design**: RESTful services with comprehensive documentation
- **UI/UX Design**: Professional web interface development
- **Testing**: Comprehensive test coverage and validation

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👨‍💻 Author

**Tushr Verma**
- **GitHub**: [@tushr23](https://github.com/tushr23)
- **Email**: tushrverma23@gmail.com
- **LinkedIn**: [Tushr Verma](https://linkedin.com/in/tushr-verma)

---

<div align="center">

**🚀 Built with ❤️ for modern AI applications**

*Showcasing enterprise-level development practices and AI integration*

[⭐ Star this repo](https://github.com/tushr23/MultimodalAssistant) | [🐛 Report Issues](https://github.com/tushr23/MultimodalAssistant/issues) | [💡 Request Features](https://github.com/tushr23/MultimodalAssistant/discussions)

</div>

## 🚀 Quick Start

**Option 1: Docker (Recommended)**
```bash
git clone <repository-url>
cd MultimodalAssistant
docker compose up --build
```

**Option 2: Local Installation**
```bash
# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR (required for text extraction)
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# macOS: brew install tesseract  
# Linux: sudo apt-get install tesseract-ocr

# Run the API server
python main.py
```

✅ Backend API: http://localhost:8000  
✅ Interactive Docs: http://localhost:8000/docs  
✅ Frontend CLI: Launches automatically  

**Note**: If Tesseract is not installed, OCR will show "OCR unavailable" but BLIP captioning still works.  

## 📋 API Example

```bash
curl -X POST "http://localhost:8000/v1/vision" \
  -F "image=@photo.jpg" \
  -F "prompt=What's in this image?"
```

**Response:**
```json
{
  "caption": "A person walking with a dog in a park",
  "ocr_text": "WELCOME TO CENTRAL PARK",
  "status": "success",
  "processing_time": 1.2
}
```

## 🧪 Testing

Run comprehensive tests with coverage analysis:

```bash
# Run all tests with coverage (100% target)
python test_api.py

# Run specific test suites
python test_final_coverage.py          # Comprehensive coverage test
python test_exception_paths.py         # Exception handling tests
python test_coverage_focused.py        # Import and constant tests

# Clean project cache
python test_api.py clean
```

**Test Coverage Features:**
- ✅ 100% line coverage target
- ✅ Comprehensive exception path testing
- ✅ Edge case validation
- ✅ Performance benchmarking
- ✅ Security validation
- ✅ Multiple image format support
- ✅ OCR functionality testing
- ✅ Model integration testing

## ⚙️ Manual Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload    # Backend
python frontend.py           # Frontend (new terminal)
```

## 🚀 Deploy

- **Railway**: Connect GitHub repo
- **Google Cloud**: `gcloud run deploy --source .`
- **AWS/Azure**: Use Docker setup

## 🔧 Features

✅ **Multi-format images**: JPEG, PNG, WebP, BMP, TIFF  
✅ **AI Captioning**: BLIP model for image understanding  
✅ **OCR Text**: Tesseract for text extraction  
✅ **Production ready**: Docker, health checks, error handling  
✅ **Comprehensive tests**: 95%+ coverage with cleanup  

## 📄 License & Credits

MIT License - **Built by [Tushr Verma](https://github.com/tushr23)** (tushrverma23@gmail.com)

---

**Portfolio Project - AI/ML & Full-Stack Development** 🎯

- 🎨 **Image Captioning** using BLIP (Bootstrapped Language-Image Pre-training)
- 📝 **Optical Character Recognition** using Tesseract OCR
- 💬 **Visual Question Answering** capabilities
- 🐳 **Containerized** with Docker Compose orchestration
- 🧪 **Comprehensive test suite** with automated cleanup
- 🔄 **CI/CD pipeline** ready for deployment

## 🚀 Features

- **Multi-format Image Support**: JPEG, PNG, WebP, BMP, TIFF
- **AI-Powered Captioning**: Advanced image understanding and description
- **Text Extraction**: OCR for documents, signs, and printed text
- **Question Answering**: Ask specific questions about image content
- **Professional API**: RESTful design with comprehensive documentation
- **Security**: Input validation, file type verification, size limits
- **Performance**: Optimized Docker builds, health checks, monitoring

## 📋 Prerequisites

- **Docker Desktop** (recommended) or **Python 3.10+**
- **4GB+ RAM** (for AI model loading)
- **1GB+ free disk space**

## 🏃‍♂️ Quick Start

### Docker Compose (Recommended)

```bash
# Clone and navigate
git clone <repository-url>
cd MultimodalAssistant

# Launch everything with one command
docker compose up --build
```

**That's it!** The system will:
1. Build optimized containers
2. Download AI models (first run only)
3. Start backend API on http://localhost:8000
4. Launch interactive frontend
5. Run health checks

### Manual Setup

Note: For local (non-Docker) runs you must have Tesseract installed on your machine and available on PATH.

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then in another terminal:
```bash
python frontend.py
```

If Tesseract isn't found locally, install it and make sure the executable is on your PATH. On Windows you may need to set:
```
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```
in your code before using pytesseract (not needed when using Docker).

## API Usage

### Using curl
```bash
curl -X POST "http://localhost:8000/v1/vision" \
  -F "image=@sample.jpg" \
  -F "prompt=Describe this image"
```

### API Reference
- Endpoint: POST `/v1/vision`
- Content-Type: `multipart/form-data`
- Form fields:
  - `image` (file, required): Image file (jpg/png/webp, etc.)
  - `prompt` (string, optional): Defaults to "Describe this image"
- Response (200):
  - `caption` (string)
  - `ocr_text` (string)
  - `prompt` (string)
  - `filename` (string)
  - `status` ("success")

### Response Format
```json
{
  "caption": "A group of people playing soccer on a field",
  "ocr_text": "Welcome to the soccer match!",
  "prompt": "Describe this image",
  "filename": "sample.jpg",
  "status": "success"
}
```

## Testing

```bash
pytest test_api.py -v
```

## Configuration
- Frontend uses the `API_URL` environment variable (defaults to `http://localhost:8000/v1/vision`). In Docker Compose it is set to `http://backend:8000/v1/vision`.

## Professional Setup Notes
- Both backend and frontend run in isolated containers
- No manual setup required—just one command
- Works on any OS and cloud platform
- Follows industry best practices for deployment and orchestration
- Full error handling and input validation
- Automated testing and CI/CD ready

## Tech Stack
- FastAPI for the backend API
- BLIP (Salesforce) for image captioning
- Tesseract for OCR
- Docker & Docker Compose for containerization
- pytest for testing

## Troubleshooting
- First run may take time to download BLIP weights.
- If running locally without Docker and OCR fails: install Tesseract and ensure it's on PATH (see Manual Setup notes).
- If Torch installation fails locally, prefer using Docker, or install a CPU-only wheel appropriate for your platform.

## 💼 Portfolio Highlights

This project demonstrates:
- **AI/ML Integration**: Computer vision and NLP deployment
- **API Development**: Professional FastAPI implementation
- **DevOps**: Docker containerization and orchestration
- **Testing**: Comprehensive test coverage with cleanup
- **Production Ready**: Security, monitoring, and deployment

Perfect for showcasing in technical interviews!

---

**Built with ❤️ for professional portfolio demonstration** 🎯

*This project showcases enterprise-level development practices and AI integration capabilities.*

### Manual Setup

Note: For local (non-Docker) runs you must have Tesseract installed on your machine and available on PATH.

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then in another terminal:
```bash
python frontend.py
```

If Tesseract isn't found locally, install it and make sure the executable is on your PATH. On Windows you may need to set:
```
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```
in your code before using pytesseract (not needed when using Docker).

## API Usage

### Using curl
```bash
curl -X POST "http://localhost:8000/v1/vision" \
  -F "image=@sample.jpg" \
  -F "prompt=Describe this image"
```

### API Reference
- Endpoint: POST `/v1/vision`
- Content-Type: `multipart/form-data`
- Form fields:
  - `image` (file, required): Image file (jpg/png/webp, etc.)
  - `prompt` (string, optional): Defaults to "Describe this image"
- Response (200):
  - `caption` (string)
  - `ocr_text` (string)
  - `prompt` (string)
  - `filename` (string)
  - `status` ("success")

### Response Format
```json
{
  "caption": "A group of people playing soccer on a field",
  "ocr_text": "Welcome to the soccer match!",
  "prompt": "Describe this image",
  "filename": "sample.jpg",
  "status": "success"
}
```

## Testing

```bash
pytest test_api.py -v
```

## Configuration
- Frontend uses the `API_URL` environment variable (defaults to `http://localhost:8000/v1/vision`). In Docker Compose it is set to `http://backend:8000/v1/vision`.

## Professional Setup Notes
- Both backend and frontend run in isolated containers
- No manual setup required—just one command
- Works on any OS and cloud platform
- Follows industry best practices for deployment and orchestration
- Full error handling and input validation
- Automated testing and CI/CD ready

## Tech Stack
- FastAPI for the backend API
- BLIP (Salesforce) for image captioning
- Tesseract for OCR
- Docker & Docker Compose for containerization
- pytest for testing

## Troubleshooting
- First run may take time to download BLIP weights.
- If running locally without Docker and OCR fails: install Tesseract and ensure it’s on PATH (see Manual Setup notes).
- If Torch installation fails locally, prefer using Docker, or install a CPU-only wheel appropriate for your platform.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👨‍💻 Credits & Contact

**Developed by**: [Tushr Verma](https://github.com/tushr23)
- **GitHub**: [@tushr23](https://github.com/tushr23)
- **Email**: tushrverma23@gmail.com
- **Portfolio**: Python Developer with Generative AI.
"# Force trigger"  
