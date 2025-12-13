# Efficient RAG with Learned Retrieval and Uncertainty Quantification

A research implementation combining differentiable retrieval gating with Bayesian uncertainty calibration for retrieval-augmented generation (RAG).

## Overview

This project implements:
- **Differentiable Retrieval Gating Network**: A learned MLP router that dynamically weights BM25 vs. dense retrieval scores per passage
- **Bayesian Confidence Calibration**: MC Dropout + Conformal Prediction for uncertainty quantification
- **Hybrid Retrieval System**: Combined BM25 (sparse) and ChromaDB (dense) retrieval with learned fusion

## Prerequisites

- Docker and Docker Compose (v2.0+)
- 8GB RAM minimum (16GB recommended)
- 10GB disk space

Verify Docker is installed:
```bash
docker --version
docker compose version
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/manikya7022/Efficient-RAG-with-Learned-Retrieval-and-Uncertainty-Quantification.git
cd Efficient-RAG-with-Learned-Retrieval-and-Uncertainty-Quantification
```

### 2. Start Docker

```bash
# macOS
open -a Docker

# Linux
sudo systemctl start docker
```

### 3. Run Setup

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

This will:
- Create necessary directories
- Start Ollama and ChromaDB services
- Pull required LLM models (llama3.2:3b, nomic-embed-text)
- Build the application Docker image

### 4. Verify Installation

```bash
docker ps
```

You should see three containers running: `rag_phd`, `chromadb_phd`, and `ollama_phd`.

## Running Experiments

### Quick Start (Automated Pipeline)

Run the complete experiment with a single command:

```bash
# Full experiment (2-4 hours)
docker-compose run --rm rag_uq bash scripts/run_experiment.sh

# Quick test mode (30 minutes)
docker-compose run --rm rag_uq bash scripts/run_experiment.sh --quick
```

### Manual Steps

#### Step 1: Prepare Corpus

Download Wikipedia articles and chunk into passages:

```bash
docker-compose run --rm rag_uq python data/preprocessing/prepare_corpus.py \
    --task all \
    --n-articles 100 \
    --n-nq 500 \
    --output-dir data
```

#### Step 2: Build Index

Create BM25 and ChromaDB indices:

```bash
docker-compose run --rm rag_uq python data/preprocessing/build_chroma_index.py \
    --corpus-path data/preprocessed/wikipedia_100k.jsonl \
    --batch-size 100
```

#### Step 3: Train Router

Train the learned routing network:

```bash
# With real data
docker-compose run --rm rag_uq python experiments/run_router_training.py \
    --nq-path data/preprocessed/nq_dev_3000.jsonl \
    --epochs 50 \
    --batch-size 16

# With synthetic data (for testing)
docker-compose run --rm rag_uq python experiments/run_router_training.py --synthetic
```

#### Step 4: Run Calibration

Build calibration set for uncertainty quantification:

```bash
docker-compose run --rm rag_uq python experiments/run_calibration.py \
    --nq-path data/preprocessed/nq_dev_3000.jsonl \
    --n-samples 500
```

#### Step 5: Run Evaluation

Execute full evaluation:

```bash
docker-compose run --rm rag_uq python experiments/run_evaluation.py \
    --nq-path data/preprocessed/nq_dev_3000.jsonl \
    --router-path models/router_lora/best_router.pt \
    --output-dir results
```

## Project Structure

```
├── rag_uq/                     # Core library
│   ├── router.py               # Learned retrieval router
│   ├── confidence.py           # MC Dropout + Conformal Prediction
│   ├── streaming_index.py      # Hybrid retrieval (BM25 + ChromaDB)
│   └── eval_protocol.py        # Evaluation metrics
├── data/
│   ├── preprocessing/          # Data pipeline scripts
│   └── preprocessed/           # Processed datasets
├── experiments/
│   ├── run_router_training.py  # Router training
│   ├── run_calibration.py      # Conformal calibration
│   └── run_evaluation.py       # Evaluation pipeline
├── models/router_lora/         # Trained model checkpoints
├── results/                    # Evaluation outputs
├── scripts/
│   ├── setup.sh                # Environment setup
│   └── run_experiment.sh       # Pipeline runner
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Output Files

After running experiments, results are stored in:

| Location | Contents |
|----------|----------|
| `models/router_lora/best_router.pt` | Trained router checkpoint |
| `models/router_lora/training_results.json` | Training metrics |
| `results/evaluation_results.json` | Evaluation metrics |
| `results/reliability_diagram.png` | Calibration plot |
| `data/preprocessed/` | Processed datasets |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama API endpoint |
| `CHROMA_HOST` | `chromadb` | ChromaDB host |
| `CHROMA_PORT` | `8000` | ChromaDB port |

### Router Hyperparameters

```python
RouterConfig(
    hidden_dim=64,      # MLP hidden dimension
    dropout=0.1,        # Dropout rate
    temperature=1.0,    # ApproxNDCG temperature
    num_layers=2        # MLP depth
)
```

## Running Tests

```bash
docker-compose run --rm rag_uq pytest tests/ -v
```

## Stopping Services

```bash
# Stop containers
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Troubleshooting

### Ollama Connection Failed

```bash
docker logs ollama_phd
docker-compose restart ollama
docker exec ollama_phd ollama pull llama3.2:3b
```

### ChromaDB Connection Failed

```bash
docker logs chromadb_phd
docker-compose restart chromadb
curl http://localhost:8000/api/v2/heartbeat
```

### Memory Issues

Reduce batch size:
```bash
docker-compose run --rm rag_uq python experiments/run_router_training.py --batch-size 8 --synthetic
```

## File Descriptions

### Core Library (`rag_uq/`)

| File | Description |
|------|-------------|
| `router.py` | Implements the `RetrievalRouter` class - a lightweight MLP that learns to weight BM25 vs dense retrieval scores. Includes `ApproxNDCGLoss` for listwise ranking optimization and `RouterTrainer` for the training loop with checkpointing. |
| `confidence.py` | Contains `MCDropoutConfidence` for uncertainty estimation via multiple LLM samples with embedding variance, and `ConformalRAG` for frequentist coverage guarantees using ROUGE-L nonconformity scores. |
| `streaming_index.py` | Implements `BM25Index` for sparse retrieval, `DenseIndex` using ChromaDB with Ollama embeddings, and `HybridRetriever` that combines both. Supports incremental indexing via `StreamingIndex`. |
| `eval_protocol.py` | Contains `RAGEvaluator` with metrics for retrieval (Recall@K, MRR, NDCG), generation (Exact Match, F1, ROUGE-L), calibration (ECE, MCE, Brier Score), and efficiency (latency, throughput). Includes reliability diagram plotting. |

### Data Pipeline (`data/preprocessing/`)

| File | Description |
|------|-------------|
| `prepare_corpus.py` | Downloads Wikipedia articles via API with rate limiting, chunks text into overlapping passages, and prepares Natural Questions dataset from HuggingFace. Supports checkpointing for resumable downloads. |
| `build_chroma_index.py` | Builds both ChromaDB (dense) and BM25 (sparse) indices from JSONL corpus files. Includes batch processing and index verification with sample queries. |
| `verify_dataset.py` | Validates JSONL datasets by computing statistics (document count, field distribution, text lengths) and checking for duplicate IDs. Generates human-readable reports. |

### Experiment Scripts (`experiments/`)

| File | Description |
|------|-------------|
| `run_router_training.py` | Loads NQ dataset, retrieves passages using hybrid retrieval, generates pseudo-relevance labels based on answer overlap, and trains the router using ApproxNDCG loss. Supports synthetic data mode for quick testing. |
| `run_calibration.py` | Runs conformal calibration by generating LLM predictions on held-out samples, computing ROUGE-L nonconformity scores against true answers, and persisting calibration data to SQLite. |
| `run_evaluation.py` | Orchestrates full RAG evaluation: performs hybrid retrieval, generates answers with LLM, computes all metrics, and produces reliability diagrams and routing analysis plots. |

### Setup Scripts (`scripts/`)

| File | Description |
|------|-------------|
| `setup.sh` | Shell script that checks Docker installation, creates directories, starts Ollama/ChromaDB services, pulls LLM models, and builds the application container. |
| `run_experiment.sh` | Automated pipeline runner that executes corpus preparation, index building, router training, calibration, and evaluation. Supports `--quick` mode for reduced samples. |
| `download_models.py` | Pre-downloads sentence-transformers models and NLTK data for offline use during Docker image build. |

### Configuration Files

| File | Description |
|------|-------------|
| `docker-compose.yml` | Defines three Docker services: `rag_uq` (application), `ollama` (LLM inference on port 11434), and `chromadb` (vector store on port 8000). |
| `Dockerfile` | Python 3.11 container with ML dependencies, copies project files, creates data directories, and pre-downloads the sentence-transformers embedding model. |
| `requirements.txt` | Python dependencies including PyTorch, transformers, ChromaDB, rank-bm25, ollama, rouge-score, and testing libraries. |
| `pyproject.toml` | Modern Python project configuration with metadata, dependencies, and tool settings for black, isort, and mypy. |
