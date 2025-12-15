# Evaluation Protocol

## Overview

The Evaluation module (`rag_uq/eval_protocol.py`) provides comprehensive metrics across:

1. **Retrieval Quality** - Recall@K, MRR, NDCG
2. **Generation Quality** - EM, F1, ROUGE-L
3. **Calibration Quality** - ECE, MCE, Brier Score
4. **Efficiency** - Latency percentiles, throughput

## Core Classes

### `RAGEvaluator`

```python
from rag_uq.eval_protocol import RAGEvaluator

evaluator = RAGEvaluator(
    output_dir="results",
    n_bins=10,
    bootstrap_samples=1000
)
```

## Retrieval Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Recall@K | `\|Retrieved ∩ Relevant\| / \|Relevant\|` | Coverage of relevant docs |
| MRR | `mean(1/rank_first_relevant)` | Ranking quality |
| NDCG@K | `DCG / IDCG` | Graded relevance |

```python
metrics = evaluator.evaluate_retrieval(
    retrieved_ids=[["d1", "d5"], ["d2", "d7"]],
    gold_ids=[["d5"], ["d2"]],
    k_values=[1, 5, 10]
)
```

## Generation Metrics

| Metric | Description |
|--------|-------------|
| Exact Match | Normalized string equality |
| Token F1 | Token overlap F1 |
| ROUGE-L | LCS-based F1 |

```python
metrics = evaluator.evaluate_generation(
    predictions=["Paris is capital", "1991"],
    references=["Paris", "1991"]
)
```

## Calibration Metrics

| Metric | Target | Meaning |
|--------|--------|---------|
| ECE | < 0.05 | Avg confidence-accuracy gap |
| MCE | < 0.10 | Max gap in any bin |
| Brier | < 0.15 | MSE of confidence |

```python
metrics = evaluator.evaluate_calibration(
    confidences=np.array([0.9, 0.7, 0.5]),
    correctness=np.array([1, 1, 0])
)
```

## Reliability Diagram

```python
evaluator.plot_reliability_diagram(
    confidences, correctness,
    save_path="results/reliability_diagram.png"
)
```

## Efficiency Metrics

```python
efficiency = evaluator.evaluate_efficiency(
    total_times=[150, 160, 145],
    router_times=[2, 3, 2],
    retrieval_times=[20, 22, 19],
    generation_times=[128, 135, 124]
)
print(f"P95: {efficiency.p95_latency_ms} ms")
```
