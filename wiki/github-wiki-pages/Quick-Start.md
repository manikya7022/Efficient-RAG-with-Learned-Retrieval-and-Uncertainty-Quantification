# Quick Start Guide

## Automated Pipeline (Recommended)

Run the complete experiment:

```bash
# Full experiment (2-4 hours)
docker-compose run --rm rag_uq bash scripts/run_experiment.sh

# Quick test mode (30 minutes)
docker-compose run --rm rag_uq bash scripts/run_experiment.sh --quick
```

## Manual Steps

### Step 1: Prepare Corpus

Download Wikipedia articles and chunk into passages:

```bash
docker-compose run --rm rag_uq python data/preprocessing/prepare_corpus.py \
    --task all \
    --n-articles 100 \
    --n-nq 500 \
    --output-dir data
```

### Step 2: Build Index

Create BM25 and ChromaDB indices:

```bash
docker-compose run --rm rag_uq python data/preprocessing/build_chroma_index.py \
    --corpus-path data/preprocessed/wikipedia_100k.jsonl \
    --batch-size 100
```

### Step 3: Train Router

```bash
# With real data
docker-compose run --rm rag_uq python experiments/run_router_training.py \
    --nq-path data/preprocessed/nq_dev_3000.jsonl \
    --epochs 50

# Quick test with synthetic data
docker-compose run --rm rag_uq python experiments/run_router_training.py --synthetic
```

### Step 4: Calibrate

```bash
docker-compose run --rm rag_uq python experiments/run_calibration.py \
    --nq-path data/preprocessed/nq_dev_3000.jsonl \
    --n-samples 500
```

### Step 5: Evaluate

```bash
docker-compose run --rm rag_uq python experiments/run_evaluation.py \
    --nq-path data/preprocessed/nq_dev_3000.jsonl \
    --router-path models/router_lora/best_router.pt \
    --output-dir results
```

## Output Files

| Location | Contents |
|----------|----------|
| `models/router_lora/best_router.pt` | Trained router |
| `results/evaluation_results.json` | Metrics |
| `results/reliability_diagram.png` | Calibration plot |

## Stopping Services

```bash
docker-compose down    # Stop containers
docker-compose down -v # Stop and remove volumes
```

## Next Steps

→ [[Configuration]] - Customize settings
→ [[Troubleshooting]] - Common issues
