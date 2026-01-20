# Multi-stage build for DocQA AI
# Stage 1: Builder - Install dependencies and build
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libopenblas-dev \
    liblapack-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    PATH=/root/.local/bin:$PATH

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd --create-home --shell /bin/bash docqa \
    && chown -R docqa:docqa /app

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder --chown=docqa:docqa /root/.local /root/.local

# Copy application code
COPY --chown=docqa:docqa . .

# Create necessary directories
RUN mkdir -p /app/data/raw \
    /app/data/processed \
    /app/data/embeddings \
    /app/data/vector_store \
    /app/logs \
    /app/cache \
    && chown -R docqa:docqa /app/data /app/logs /app/cache

# Expose ports
EXPOSE 8000 9090

# Switch to non-root user
USER docqa

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
