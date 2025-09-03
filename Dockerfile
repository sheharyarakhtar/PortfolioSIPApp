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

# Expose the port Streamlit runs on
EXPOSE 1000

# Health check
HEALTHCHECK CMD curl --fail http://localhost:1000/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "main.py", "--server.port=1000", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
