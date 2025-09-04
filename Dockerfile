FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only runtime libraries (lighter than full build toolchain)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (prefer wheels, skip building from source for speed)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --only-binary=:all: -r requirements.txt

# Copy the application code
COPY . .

# Health check (Render sets $PORT, so reference it)
HEALTHCHECK CMD curl --fail http://localhost:$PORT/_stcore/health || exit 1

# Expose port (Render ignores EXPOSE, but it’s good practice)
EXPOSE 10000

# Run the app (optimized for Render free tier - fast startup)
CMD streamlit run main.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.headless=true \
    --server.fileWatcherType=none
