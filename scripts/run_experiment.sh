#!/bin/bash
# Run full experiment pipeline
# Usage: ./scripts/run_experiment.sh [--quick]

set -e

QUICK_MODE=false
if [ "$1" == "--quick" ]; then
    QUICK_MODE=true
    echo "Running in quick mode (reduced samples)"
fi

echo "=========================================="
echo "RAG-UQ Experiment Pipeline"
echo "=========================================="

# Configuration
if [ "$QUICK_MODE" = true ]; then
    N_ARTICLES=100
    N_NQ=500
    N_CALIBRATION=100
    N_TEST=50
    EPOCHS=10
else
    N_ARTICLES=1000
    N_NQ=3000
    N_CALIBRATION=500
    N_TEST=200
    EPOCHS=50
fi

# Step 1: Check if corpus exists
if [ ! -f "data/preprocessed/wikipedia_100k.jsonl" ]; then
    echo ""
    echo "[1/5] Preparing Wikipedia corpus..."
    python data/preprocessing/prepare_corpus.py \
        --task download \
        --n-articles $N_ARTICLES
    
    python data/preprocessing/prepare_corpus.py \
        --task chunk
fi

# Step 2: Check if NQ dataset exists
if [ ! -f "data/preprocessed/nq_dev_3000.jsonl" ]; then
    echo ""
    echo "[2/5] Preparing Natural Questions dataset..."
    python data/preprocessing/prepare_corpus.py \
        --task nq \
        --n-nq $N_NQ
fi

# Step 3: Build index
echo ""
echo "[3/5] Building retrieval index..."
python data/preprocessing/build_chroma_index.py \
    --corpus data/preprocessed/wikipedia_100k.jsonl

# Step 4: Train router
echo ""
echo "[4/5] Training retrieval router..."
python experiments/run_router_training.py \
    --epochs $EPOCHS \
    --batch-size 16

# Step 5: Run calibration
echo ""
echo "[5/5] Running conformal calibration..."
python experiments/run_calibration.py \
    --n-samples $N_CALIBRATION

# Step 6: Evaluate
echo ""
echo "[6/6] Running evaluation..."
python experiments/run_evaluation.py \
    --n-samples $N_TEST

echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo ""
echo "Results saved to: results/"
echo "  - evaluation_results.json"
echo "  - reliability_diagram.png"
echo "  - routing_analysis.png"
echo ""
echo "Model saved to: models/router_lora/"
