# System Architecture

## Overview

The Efficient RAG system implements a novel approach to Retrieval-Augmented Generation that combines:

1. **Learned Retrieval Gating** - A differentiable router that learns optimal weighting between sparse and dense retrieval
2. **Bayesian Uncertainty Quantification** - MC Dropout + Conformal Prediction for calibrated confidence estimates
3. **Hybrid Retrieval** - Combined BM25 (lexical) and ChromaDB (semantic) retrieval

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Query Processing                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌─────────┐                                                              │
│    │  Query  │                                                              │
│    └────┬────┘                                                              │
│         │                                                                    │
│         ▼                                                                    │
│    ┌─────────────────────────────────────────────────────────┐              │
│    │              Hybrid Retriever                            │              │
│    │  ┌─────────────────┐    ┌─────────────────┐             │              │
│    │  │   BM25 Index    │    │  Dense Index    │             │              │
│    │  │   (Sparse)      │    │  (ChromaDB)     │             │              │
│    │  └────────┬────────┘    └────────┬────────┘             │              │
│    │           │                      │                       │              │
│    │           ▼                      ▼                       │              │
│    │      bm25_scores           dense_scores                  │              │
│    └───────────┬─────────────────────┬────────────────────────┘              │
│                │                      │                                      │
│                └──────────┬───────────┘                                      │
│                           ▼                                                  │
│              ┌────────────────────────┐                                      │
│              │   Retrieval Router     │                                      │
│              │   (Learned MLP)        │                                      │
│              │                        │                                      │
│              │  w = σ(MLP(bm25, d))   │                                      │
│              │  s = w·d + (1-w)·bm25  │                                      │
│              └───────────┬────────────┘                                      │
│                          │                                                   │
│                          ▼                                                   │
│              ┌────────────────────────┐                                      │
│              │   Top-K Passages       │                                      │
│              └───────────┬────────────┘                                      │
│                          │                                                   │
└──────────────────────────┼──────────────────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────────────────────┐
│                          │           Generation + UQ                        │
├──────────────────────────┼──────────────────────────────────────────────────┤
│                          ▼                                                   │
│              ┌────────────────────────┐                                      │
│              │   LLM Generation       │                                      │
│              │   (Ollama/llama3.2)    │                                      │
│              └───────────┬────────────┘                                      │
│                          │                                                   │
│         ┌────────────────┼────────────────┐                                 │
│         ▼                ▼                ▼                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                         │
│  │ MC Dropout   │ │ Conformal    │ │ Retrieval    │                         │
│  │ Confidence   │ │ Prediction   │ │ Uncertainty  │                         │
│  │              │ │              │ │              │                         │
│  │ Embedding    │ │ ROUGE-L      │ │ Score        │                         │
│  │ Variance     │ │ Nonconform.  │ │ Variance     │                         │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘                         │
│         │                │                │                                  │
│         └────────────────┼────────────────┘                                  │
│                          ▼                                                   │
│              ┌────────────────────────┐                                      │
│              │   Combined Confidence  │                                      │
│              │   + Coverage Guarantee │                                      │
│              └───────────┬────────────┘                                      │
│                          │                                                   │
└──────────────────────────┼──────────────────────────────────────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │ Final Answer   │
                  │ + Confidence   │
                  │ + Reliability  │
                  └────────────────┘
```

## Component Overview

### 1. Hybrid Retriever (`streaming_index.py`)

The retrieval layer combines two complementary approaches:

| Component | Type | Purpose | Library |
|-----------|------|---------|---------|
| `BM25Index` | Sparse | Lexical/keyword matching | rank-bm25 |
| `DenseIndex` | Dense | Semantic similarity | ChromaDB + Ollama embeddings |
| `HybridRetriever` | Combined | Unified interface | Custom |

**Key Features:**
- Incremental indexing without full rebuild
- Persistence support for both indices
- Batch processing for memory efficiency

### 2. Retrieval Router (`router.py`)

A learned MLP that outputs per-passage weights for combining retrieval scores:

```python
# Per-passage gating weight
w = sigmoid(MLP([bm25_score, dense_score]))

# Hybrid score
hybrid_score = w * dense_score + (1 - w) * bm25_score
```

**Training:**
- Loss: ApproxNDCG (differentiable ranking loss)
- Labels: Pseudo-relevance from answer overlap
- Architecture: 2-layer MLP with dropout

### 3. Uncertainty Quantification (`confidence.py`)

Two complementary approaches:

| Method | Type | Provides |
|--------|------|----------|
| MC Dropout | Bayesian | Epistemic uncertainty via sampling |
| Conformal Prediction | Frequentist | Coverage guarantees |

**MC Dropout Implementation:**
- Uses temperature/top-p variation as dropout proxy
- Computes embedding-space variance of samples
- Returns consensus answer closest to centroid

**Conformal Prediction:**
- Nonconformity score: 1 - ROUGE-L
- Calibration via held-out set
- Distribution-free coverage guarantee

### 4. Evaluation Protocol (`eval_protocol.py`)

Comprehensive metrics across four categories:

| Category | Metrics |
|----------|---------|
| Retrieval | Recall@K, MRR, NDCG@10, Precision@K |
| Generation | Exact Match, Token F1, ROUGE-L |
| Calibration | ECE, MCE, Brier Score |
| Efficiency | Latency (p50/p95/p99), Throughput |

## Data Flow

1. **Query Input** → Tokenized for both retrievers
2. **Parallel Retrieval** → BM25 and Dense scores computed
3. **Router** → Learns optimal combination weights
4. **Context Formation** → Top-K passages concatenated
5. **LLM Generation** → Answer generated with context
6. **UQ Sampling** → Multiple samples for uncertainty
7. **Calibration** → Conformal threshold applied
8. **Output** → Answer + confidence + reliability flag

## External Dependencies

| Service | Purpose | Port |
|---------|---------|------|
| Ollama | LLM inference + embeddings | 11434 |
| ChromaDB | Vector storage | 8000 |

## Directory Structure

```
rag_uq/
├── __init__.py           # Package exports
├── router.py             # Learned routing network
├── confidence.py         # MC Dropout + Conformal
├── streaming_index.py    # Hybrid retrieval
└── eval_protocol.py      # Evaluation metrics
```
