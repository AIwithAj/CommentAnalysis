# ============================================================
# Stage 1: Base Image (Slim + Build Essentials)
# ============================================================
FROM python:3.10-slim AS base


# Working directory
WORKDIR /app

# System dependencies (lightgbm + nltk need these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Stage 2: Install Python dependencies
# ============================================================
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download essential NLTK data
RUN python -m nltk.downloader stopwords wordnet

# ============================================================
# Stage 3: Copy app and expose port
# ============================================================
COPY app.py /app/

# Expose Flask/Gunicorn port
EXPOSE 8000

# Default environment variables (can override in CI/CD)

# ============================================================
# Stage 4: Production Command
# ============================================================
# Use Gunicorn for production server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "app:app"]
