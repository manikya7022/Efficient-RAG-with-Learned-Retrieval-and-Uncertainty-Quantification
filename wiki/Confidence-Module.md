# Confidence Module

## Overview

The Confidence module (`rag_uq/confidence.py`) implements uncertainty quantification for RAG using two complementary approaches:

1. **MC Dropout Confidence** - Bayesian approach using sampling
2. **Conformal Prediction** - Frequentist approach with coverage guarantees

## Core Classes

### `ConfidenceResult`

Result dataclass from MC Dropout confidence estimation:

```python
@dataclass
class ConfidenceResult:
    answers: List[str]              # All sampled answers
    consensus_answer: str           # Best answer (closest to centroid)
    uncertainty_score: float        # Embedding variance (0-1)
    confidence: float               # 1 - uncertainty
    embedding_variance: float       # Raw variance in embedding space
    lexical_diversity: float        # Type-token ratio
    metadata: Dict[str, Any]        # Additional info
```

### `ConformalResult`

Result dataclass from conformal prediction:

```python
@dataclass  
class ConformalResult:
    prediction: str          # Generated answer
    confidence: float        # Estimated conformity
    p_value: float           # Conformal p-value
    is_reliable: bool        # Below threshold?
    coverage_alpha: float    # Target miscoverage rate
    metadata: Dict[str, Any]
```

---

## MC Dropout Confidence

### Concept

Since Ollama doesn't support native dropout, we simulate epistemic uncertainty through:

1. **Temperature variation** - Sample from broader distributions
2. **Top-p variation** - Add stochasticity to sampling
3. **Multiple samples** - Generate N answers per query

### Class: `MCDropoutConfidence`

```python
from rag_uq.confidence import MCDropoutConfidence

mc_conf = MCDropoutConfidence(
    llm_client=ollama_client,
    n_samples=10,                      # Number of MC samples
    embedding_model='all-MiniLM-L6-v2', # For variance computation
    temperature_range=(0.5, 1.2),      # Sampling variation
    top_p_range=(0.8, 0.95),
    max_tokens=100
)
```

### Key Method: `get_confidence_interval`

```python
result = mc_conf.get_confidence_interval(
    prompt="You are a helpful assistant...",
    context="Retrieved passages here...",
    question="What is the capital of France?",
    model="llama3.2:3b"
)

print(f"Answer: {result.consensus_answer}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Uncertainty: {result.uncertainty_score:.3f}")
```

### How It Works

```
Query + Context
      │
      ▼
┌─────────────────────────────────┐
│  Generate N samples with        │
│  varying temp/top_p             │
│                                 │
│  Sample 1: T=0.6, p=0.85 → A₁   │
│  Sample 2: T=0.9, p=0.90 → A₂   │
│  Sample 3: T=1.1, p=0.82 → A₃   │
│  ...                            │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  Embed all answers              │
│  e₁ = embed(A₁)                 │
│  e₂ = embed(A₂)                 │
│  ...                            │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  Compute centroid & variance    │
│                                 │
│  centroid = mean(e₁, e₂, ...)   │
│  variance = mean(‖eᵢ - c‖²)     │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  Select consensus answer        │
│                                 │
│  A* = argmin ‖eᵢ - centroid‖    │
└─────────────────────────────────┘
      │
      ▼
   (A*, confidence, uncertainty)
```

### Uncertainty Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Embedding Variance | `mean(‖eᵢ - centroid‖)` | High = inconsistent answers |
| Lexical Diversity | `unique_tokens / total_tokens` | High = varied vocabulary |
| Confidence | `1 - normalized_variance` | High = consistent, reliable |

---

## Conformal Prediction

### Concept

Conformal prediction provides **distribution-free coverage guarantees**:

> With probability ≥ 1-α, the true answer is captured by our prediction set.

### Class: `ConformalRAG`

```python
from rag_uq.confidence import ConformalRAG

conformal = ConformalRAG(
    llm_client=ollama_client,
    calibration_db_path="data/calibration_scores.db",
    alpha=0.1  # 90% coverage target
)
```

### Calibration Phase

Build calibration set from held-out data:

```python
stats = conformal.calibrate(
    questions=["Q1", "Q2", ...],
    contexts=["C1", "C2", ...],
    true_answers=["A1", "A2", ...],
    model="llama3.2:3b",
    skip_existing=True  # Resume from previous
)

print(f"Calibrated on {stats['n_samples']} samples")
print(f"Mean nonconformity: {stats['mean_score']:.3f}")
print(f"Threshold (α={0.1}): {stats['threshold']:.3f}")
```

### Prediction Phase

```python
result = conformal.predict_with_coverage(
    question="What year was Python created?",
    context="Python was created by Guido van Rossum...",
    model="llama3.2:3b"
)

print(f"Prediction: {result.prediction}")
print(f"Reliable: {result.is_reliable}")
print(f"P-value: {result.p_value:.3f}")
```

### Nonconformity Score

The nonconformity score measures how "different" a prediction is:

```
s(q, a) = 1 - ROUGE-L(prediction, true_answer)
```

- **s = 0**: Perfect match (most conforming)
- **s = 1**: No overlap (least conforming)

### Coverage Guarantee

Given calibration set of size n:

```
threshold = quantile(scores, (n+1)(1-α) / n)
```

**Theorem**: For exchangeable data:
```
P(s_new ≤ threshold) ≥ 1 - α
```

### Database Schema

Calibration scores are persisted in SQLite:

```sql
CREATE TABLE calibration_scores (
    id INTEGER PRIMARY KEY,
    query_hash TEXT UNIQUE,
    question TEXT,
    context TEXT,
    prediction TEXT,
    true_answer TEXT,
    nonconformity_score REAL,
    timestamp DATETIME
)
```

---

## Combined Uncertainty

For production use, combine both approaches:

```python
from rag_uq.confidence import MCDropoutConfidence, ConformalRAG

# Initialize both
mc_conf = MCDropoutConfidence(llm_client, n_samples=10)
conformal = ConformalRAG(llm_client, alpha=0.1)

# Get MC Dropout estimate
mc_result = mc_conf.get_confidence_interval(prompt, context, question)

# Get Conformal coverage
cf_result = conformal.predict_with_coverage(question, context)

# Combined decision
final_confidence = (mc_result.confidence + cf_result.confidence) / 2
is_reliable = mc_result.confidence > 0.7 and cf_result.is_reliable
```

### Combined Uncertainty Formula

```
U_total = λ₁·U_retrieval + λ₂·U_MC + λ₃·(1 - conf_conformal)
```

Where λ₁ + λ₂ + λ₃ = 1 (typically λ₁=0.2, λ₂=0.4, λ₃=0.4).

---

## Best Practices

### When to Use MC Dropout

- ✅ Need relative uncertainty ranking
- ✅ Fast inference required (fewer samples OK)
- ✅ Interpretable embedding-space analysis

### When to Use Conformal Prediction

- ✅ Need formal coverage guarantees
- ✅ Have labeled calibration data
- ✅ High-stakes decisions requiring reliability

### Recommended Settings

| Use Case | MC Samples | Alpha | Notes |
|----------|------------|-------|-------|
| Quick inference | 5 | 0.2 | Speed priority |
| Balanced | 10 | 0.1 | Default recommendation |
| High reliability | 20 | 0.05 | Safety-critical apps |
