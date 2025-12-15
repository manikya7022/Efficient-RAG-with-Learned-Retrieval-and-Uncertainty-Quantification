# Efficient RAG with Learned Retrieval and Uncertainty Quantification

Welcome to the project wiki! This documentation covers installation, usage, and tutorials.

## Quick Start

1. **Prerequisites**: Docker & Docker Compose v2.0+
2. **Clone**: `git clone https://github.com/manikya7022/Efficient-RAG-with-Learned-Retrieval-and-Uncertainty-Quantification.git`
3. **Setup**: `./scripts/setup.sh`
4. **Run**: `docker-compose run --rm rag_uq bash scripts/run_experiment.sh`

## Documentation

### User Guide
- [[Installation]] - Complete setup instructions
- [[Quick-Start]] - Run your first experiment
- [[Configuration]] - Environment variables and settings

### Technical Reference
- [[Architecture]] - System design and components
- [[API-Reference]] - Complete API documentation

### Advanced Topics
- [[Uncertainty-Theory]] - Mathematical foundations
- [[Troubleshooting]] - Common issues and solutions

## Project Overview

This project implements:
- **Differentiable Retrieval Router** - Learns optimal BM25 vs. dense weighting
- **Bayesian Uncertainty Quantification** - MC Dropout + Conformal Prediction
- **Hybrid Retrieval System** - BM25 + ChromaDB with learned fusion

## Links

- [GitHub Repository](https://github.com/manikya7022/Efficient-RAG-with-Learned-Retrieval-and-Uncertainty-Quantification)
- [Technical Documentation](https://github.com/manikya7022/Efficient-RAG-with-Learned-Retrieval-and-Uncertainty-Quantification/tree/main/wiki) (in-repo)
