# Base image with official slim Python 3.12
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies (compiler tools + MySQL client headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    default-libmysqlclient-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source files
COPY src/ ./src/
COPY api/ ./api/
COPY database/ ./database/
COPY models/ ./models/
COPY dataset/ ./dataset/
COPY reports/ ./reports/
COPY sql/ ./sql/
COPY .env.example ./.env.example

# Expose FastAPI port
EXPOSE 8000

# Health check via /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn web server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
