# Multi-stage build for production optimization with pinned base images
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage  
FROM python:3.10-slim

# Minimal runtime deps
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/app/.local

# Set up application directory
WORKDIR /app
RUN chown -R app:app /app

# Copy application code
COPY --chown=app:app main.py test_main.py pyproject.toml .flake8 ./

# Switch to non-root user
USER app

# Ensure scripts in .local are usable
ENV PATH=/home/app/.local/bin:$PATH
ENV PYTHONPATH=/app

# Health check (enhanced)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application with better production settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--access-log"]
