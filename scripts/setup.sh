#!/bin/bash
# Setup script for RAG with Learned Retrieval and Uncertainty Quantification
# This script sets up the complete development environment

set -e

echo "=========================================="
echo "RAG-UQ Environment Setup"
echo "=========================================="

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed. Please install it first."
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data/preprocessed data/raw models/router_lora results

# Start Docker services
echo "Starting Docker services..."
docker-compose up -d ollama chromadb

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 15

# Pull Ollama models
echo "Pulling Ollama models..."
docker exec ollama_phd ollama pull llama3.2:3b || echo "Warning: Failed to pull llama3.2:3b"
docker exec ollama_phd ollama pull nomic-embed-text || echo "Warning: Failed to pull nomic-embed-text"

# Build the main container
echo "Building RAG-UQ container..."
docker-compose build rag_uq

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Prepare corpus: docker-compose run --rm rag_uq python data/preprocessing/prepare_corpus.py --n-articles 1000"
echo "2. Build index: docker-compose run --rm rag_uq python data/preprocessing/build_chroma_index.py"
echo "3. Train router: docker-compose run --rm rag_uq python experiments/run_router_training.py"
echo "4. Run evaluation: docker-compose run --rm rag_uq python experiments/run_evaluation.py"
echo ""
echo "For interactive shell: docker-compose run --rm rag_uq bash"
