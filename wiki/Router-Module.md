# Router Module

## Overview

The Retrieval Router (`rag_uq/router.py`) implements a differentiable gating network that learns to optimally weight BM25 vs. dense retrieval scores on a per-passage basis.

## Core Classes

### `RouterConfig`

Configuration dataclass for the router:

```python
@dataclass
class RouterConfig:
    hidden_dim: int = 64       # MLP hidden layer size
    dropout: float = 0.1       # Dropout rate for regularization
    temperature: float = 1.0   # ApproxNDCG temperature
    num_layers: int = 2        # Number of MLP layers
    use_batch_norm: bool = False
```

### `RetrievalRouter`

The main router network implementing learned score fusion.

#### Architecture

```
Input: [bm25_score, dense_score] (2 features per passage)
   │
   ▼
Linear(2 → hidden_dim)
   │
   ▼
ReLU + Dropout
   │
   ▼
Linear(hidden_dim → hidden_dim)  # if num_layers > 1
   │
   ▼
ReLU + Dropout
   │
   ▼
Linear(hidden_dim → 1)
   │
   ▼
Sigmoid → weight ∈ [0, 1]
```

#### Key Methods

| Method | Description |
|--------|-------------|
| `forward(bm25_scores, dense_scores)` | Compute per-passage gating weights |
| `hybrid_rerank(bm25_scores, dense_scores, top_k)` | Combine scores and return top-K |
| `get_routing_decision(bm25_scores, dense_scores, threshold)` | Interpretable routing stats |

#### Example Usage

```python
from rag_uq.router import RetrievalRouter, RouterConfig

# Initialize
config = RouterConfig(hidden_dim=64, dropout=0.1)
router = RetrievalRouter(config)

# Forward pass
# bm25_scores: [batch_size, num_passages]
# dense_scores: [batch_size, num_passages]
weights = router(bm25_scores, dense_scores)

# Hybrid reranking
top_scores, top_indices = router.hybrid_rerank(
    bm25_scores, 
    dense_scores, 
    top_k=10
)
```

### `ApproxNDCGLoss`

Differentiable approximation to NDCG for listwise learning-to-rank.

#### Mathematical Formulation

**Approximate Ranks:**
```
rank_i ≈ 1 + Σ_j σ((s_j - s_i) / τ)
```
where τ is temperature and σ is sigmoid.

**ApproxNDCG:**
```
ApproxNDCG = DCG / IDCG

DCG = Σ_i (2^rel_i - 1) / log₂(rank_i + 1)
```

#### Usage

```python
from rag_uq.router import ApproxNDCGLoss

loss_fn = ApproxNDCGLoss(temperature=1.0)

# predicted_scores: [batch_size, num_items]
# relevance_labels: [batch_size, num_items]
loss = loss_fn(predicted_scores, relevance_labels)
```

### `RouterTrainer`

Training loop with checkpointing and early stopping.

#### Features

- Automatic checkpoint saving
- Early stopping with patience
- Validation metrics tracking
- CPU-optimized training

#### Usage

```python
from rag_uq.router import RetrievalRouter, RouterTrainer

router = RetrievalRouter()
trainer = RouterTrainer(
    router=router,
    learning_rate=1e-3,
    weight_decay=1e-4,
    checkpoint_dir="models/router_lora"
)

# Train
results = trainer.train(
    train_data=(bm25_train, dense_train, labels_train),
    val_data=(bm25_val, dense_val, labels_val),
    num_epochs=50,
    batch_size=16,
    early_stopping_patience=10
)

# Save/Load
trainer.save_checkpoint("models/router_lora/best_router.pt")
trainer.load_checkpoint("models/router_lora/best_router.pt")
```

## Training Pipeline

### 1. Data Preparation

Training data is created by:
1. Running queries through both retrievers
2. Computing pseudo-relevance labels based on answer overlap

```python
# Pseudo-relevance: does passage contain the answer?
relevance = 1.0 if answer.lower() in passage.lower() else 0.0
```

### 2. Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        bm25, dense, labels = batch
        
        # Forward pass
        weights = router(bm25, dense)
        hybrid_scores = weights * dense + (1 - weights) * bm25
        
        # Compute loss
        loss = approx_ndcg_loss(hybrid_scores, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 3. Evaluation

The router is evaluated on:
- Retrieval metrics (Recall@K, MRR, NDCG)
- Router accuracy (how often it chooses correctly)
- Ablation vs. BM25-only and Dense-only baselines

## Score Normalization

The router normalizes input scores for stable training:

```python
def _normalize_scores(self, bm25_scores, dense_scores):
    # Running mean/std for inference stability
    bm25_norm = (bm25_scores - self.bm25_mean) / self.bm25_std
    dense_norm = (dense_scores - self.dense_mean) / self.dense_std
    return bm25_norm, dense_norm
```

## Interpretability

The `get_routing_decision` method provides insights:

```python
decision = router.get_routing_decision(bm25_scores, dense_scores)
print(decision)
# {
#   'mean_weight': 0.65,           # Average toward dense (>0.5)
#   'dense_preferred': 0.72,       # 72% passages prefer dense
#   'bm25_preferred': 0.28,        # 28% passages prefer BM25
#   'weight_variance': 0.12,       # Low variance = consistent
#   'per_passage_weights': [...]   # Individual weights
# }
```

## Hyperparameter Tuning

| Parameter | Range | Impact |
|-----------|-------|--------|
| `hidden_dim` | 32-128 | Capacity vs. overfitting |
| `dropout` | 0.0-0.3 | Regularization strength |
| `temperature` | 0.5-2.0 | Gradient smoothness |
| `learning_rate` | 1e-4 - 1e-2 | Convergence speed |
| `num_layers` | 1-3 | Model complexity |

## Theoretical Background

The router approximates the oracle decision:

```
w*(p) = 𝟙[s_dense(p) > s_bm25(p) for relevant p]
```

By learning from pseudo-relevance labels, the router discovers when:
- **Dense is better**: Semantic/paraphrase queries
- **BM25 is better**: Exact keyword matching, rare terms
