# API Reference

## Package: `rag_uq`

### Quick Import

```python
from rag_uq import (
    RetrievalRouter, RouterConfig, RouterTrainer,
    MCDropoutConfidence, ConformalRAG,
    HybridRetriever, BM25Index, DenseIndex,
    RAGEvaluator
)
```

---

## `router.py`

### `RouterConfig`
```python
RouterConfig(
    hidden_dim: int = 64,
    dropout: float = 0.1,
    temperature: float = 1.0,
    num_layers: int = 2,
    use_batch_norm: bool = False
)
```

### `RetrievalRouter`
```python
router = RetrievalRouter(config: RouterConfig = None)

# Methods
router.forward(bm25_scores, dense_scores) → weights
router.hybrid_rerank(bm25_scores, dense_scores, top_k=10) → (scores, indices)
router.get_routing_decision(bm25_scores, dense_scores, threshold=0.5) → dict
```

### `ApproxNDCGLoss`
```python
loss_fn = ApproxNDCGLoss(temperature: float = 1.0)
loss = loss_fn(predicted_scores, relevance_labels, mask=None)
```

### `RouterTrainer`
```python
trainer = RouterTrainer(router, learning_rate=1e-3, weight_decay=1e-4)
trainer.train(train_data, val_data, num_epochs=50, batch_size=16)
trainer.save_checkpoint(path)
trainer.load_checkpoint(path)
```

---

## `confidence.py`

### `MCDropoutConfidence`
```python
mc = MCDropoutConfidence(
    llm_client,
    n_samples: int = 10,
    embedding_model: str = 'all-MiniLM-L6-v2',
    temperature_range: tuple = (0.5, 1.2),
    top_p_range: tuple = (0.8, 0.95)
)

result = mc.get_confidence_interval(prompt, context, question, model)
# Returns: ConfidenceResult
```

### `ConformalRAG`
```python
conformal = ConformalRAG(
    llm_client,
    calibration_db_path: str = "data/calibration_scores.db",
    alpha: float = 0.1
)

conformal.calibrate(questions, contexts, true_answers, model)
result = conformal.predict_with_coverage(question, context, model)
# Returns: ConformalResult
```

---

## `streaming_index.py`

### `Document`
```python
doc = Document(id="doc1", text="...", title="...", metadata={})
```

### `BM25Index`
```python
bm25 = BM25Index(persist_path=None, k1=1.5, b=0.75)
bm25.add_documents(docs: List[Document])
results = bm25.search(query, top_k=10)  # [(doc_id, score), ...]
```

### `DenseIndex`
```python
dense = DenseIndex(
    collection_name="rag_documents",
    chroma_host="chromadb",
    chroma_port=8000
)
dense.add_documents(docs, batch_size=100)
results = dense.search(query, top_k=10)  # [(id, score, text), ...]
```

### `HybridRetriever`
```python
hybrid = HybridRetriever(bm25_persist_path, chroma_collection, ...)
hybrid.add_documents(docs)
results = hybrid.retrieve(query, top_k=10, alpha=0.5)
bm25, dense, ids, texts = hybrid.get_scores_for_router(query, num_passages=20)
```

---

## `eval_protocol.py`

### `RAGEvaluator`
```python
evaluator = RAGEvaluator(output_dir="results", n_bins=10)

# Retrieval
metrics = evaluator.evaluate_retrieval(retrieved_ids, gold_ids, k_values=[1,5,10])

# Generation
metrics = evaluator.evaluate_generation(predictions, references)

# Calibration
metrics = evaluator.evaluate_calibration(confidences, correctness)

# Efficiency
metrics = evaluator.evaluate_efficiency(total_times, router_times, ...)

# Plotting
evaluator.plot_reliability_diagram(confidences, correctness, save_path)
```

---

## Return Types

| Class | Key Fields |
|-------|------------|
| `ConfidenceResult` | `consensus_answer`, `confidence`, `uncertainty_score` |
| `ConformalResult` | `prediction`, `is_reliable`, `p_value` |
| `RetrievalResult` | `doc_id`, `text`, `bm25_score`, `dense_score` |
| `RetrievalMetrics` | `recall_at_k`, `mrr`, `ndcg_at_10` |
| `GenerationMetrics` | `exact_match`, `f1`, `rouge_l` |
| `CalibrationMetrics` | `ece`, `mce`, `brier_score` |
| `EfficiencyMetrics` | `avg_latency_ms`, `p95_latency_ms`, `throughput_qps` |
