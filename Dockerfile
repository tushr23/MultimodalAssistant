# Multi-stage build for production optimization
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10-slim

# Install runtime dependencies and clean up in single layer
RUN apt-get update && apt-get install -y \
    curl \
    libtesseract-dev \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security and cache directory
RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /tmp/huggingface_cache \
    && chown -R app:app /tmp/huggingface_cache

# Copy Python packages from builder
COPY --from=builder /root/.local /home/app/.local

# Set up application directory
WORKDIR /app
RUN chown -R app:app /app

# Copy application code
COPY --chown=app:app . .

# Switch to non-root user
USER app

# Ensure scripts in .local are usable
ENV PATH=/home/app/.local/bin:$PATH
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
