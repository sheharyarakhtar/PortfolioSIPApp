FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including build tools for cvxpy/osqp
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Health check
HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1


# Run the application
EXPOSE 10000

CMD streamlit run main.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
