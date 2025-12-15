# Installation Guide

## Prerequisites

- **Docker** and **Docker Compose** (v2.0+)
- 8GB RAM minimum (16GB recommended)
- 10GB disk space

Verify Docker:
```bash
docker --version
docker compose version
```

## Step 1: Clone Repository

```bash
git clone https://github.com/manikya7022/Efficient-RAG-with-Learned-Retrieval-and-Uncertainty-Quantification.git
cd Efficient-RAG-with-Learned-Retrieval-and-Uncertainty-Quantification
```

## Step 2: Start Docker

**macOS:**
```bash
open -a Docker
```

**Linux:**
```bash
sudo systemctl start docker
```

## Step 3: Run Setup

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

This will:
- Create necessary directories
- Start Ollama and ChromaDB services
- Pull LLM models (llama3.2:3b, nomic-embed-text)
- Build the application Docker image

## Step 4: Verify Installation

```bash
docker ps
```

You should see three containers:
- `rag_phd` - Main application
- `chromadb_phd` - Vector database
- `ollama_phd` - LLM inference

## Environment Variables

Copy the example env file:
```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama API |
| `CHROMA_HOST` | `chromadb` | ChromaDB host |
| `CHROMA_PORT` | `8000` | ChromaDB port |

## Next Steps

→ [[Quick-Start]] - Run your first experiment
