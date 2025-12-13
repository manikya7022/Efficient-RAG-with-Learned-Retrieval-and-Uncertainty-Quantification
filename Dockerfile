FROM python:3.11-slim

# Install system dependencies for CPU ML
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/preprocessed /app/data/raw /app/models/router_lora /app/results

# Download small models on build (sentence-transformers)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["bash"]
