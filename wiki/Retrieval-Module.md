# Retrieval Module

## Overview

The Retrieval module (`rag_uq/streaming_index.py`) implements a hybrid retrieval system combining:

- **BM25** - Sparse lexical matching (rank-bm25)
- **Dense** - Semantic similarity via embeddings (ChromaDB + Ollama)

## Core Classes

### `Document`

Data class representing an indexable document:

```python
@dataclass
class Document:
    id: str                           # Unique identifier
    text: str                         # Document content
    title: Optional[str] = None       # Optional title
    metadata: Optional[Dict] = None   # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]: ...
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Document': ...
```

### `RetrievalResult`

Result from hybrid retrieval:

```python
@dataclass
class RetrievalResult:
    doc_id: str                       # Document ID
    text: str                         # Document text
    bm25_score: float                 # BM25 relevance score
    dense_score: float                # Dense similarity score
    hybrid_score: Optional[float]     # Combined score (if routed)
    title: Optional[str] = None
    metadata: Optional[Dict] = None
```

---

## BM25 Index

### Class: `BM25Index`

Sparse retrieval using BM25Okapi algorithm.

```python
from rag_uq.streaming_index import BM25Index, Document

# Initialize with persistence
bm25 = BM25Index(
    persist_path="data/bm25_index",
    k1=1.5,    # Term frequency saturation
    b=0.75     # Length normalization
)

# Add documents
docs = [
    Document(id="doc1", text="Python is a programming language"),
    Document(id="doc2", text="Machine learning uses Python")
]
bm25.add_documents(docs)

# Search
results = bm25.search("Python programming", top_k=10)
# Returns: [(doc_id, score), ...]
```

### BM25 Algorithm

```
score(D, Q) = Σ IDF(qᵢ) · (f(qᵢ, D) · (k₁ + 1)) / (f(qᵢ, D) + k₁ · (1 - b + b · |D|/avgdl))
```

Where:
- `f(qᵢ, D)` = frequency of term qᵢ in document D
- `|D|` = document length
- `avgdl` = average document length
- `k₁` = term frequency saturation (default 1.5)
- `b` = length normalization (default 0.75)

### Features

| Feature | Description |
|---------|-------------|
| Persistence | Saves index to disk for fast reload |
| Incremental | Add documents without full rebuild |
| Fast | O(query_terms × avg_doc_length) search |

---

## Dense Index

### Class: `DenseIndex`

Semantic retrieval using ChromaDB with Ollama embeddings.

```python
from rag_uq.streaming_index import DenseIndex, Document

# Initialize with ChromaDB
dense = DenseIndex(
    collection_name="rag_documents",
    persist_directory="./data/chroma_db",
    embedding_model="nomic-embed-text",
    chroma_host="chromadb",  # Docker service name
    chroma_port=8000
)

# Add documents (batched for efficiency)
dense.add_documents(docs, batch_size=100)

# Search
results = dense.search("semantic meaning query", top_k=10)
# Returns: [(doc_id, score, text), ...]
```

### Embedding Process

```
Document Text
     │
     ▼
┌─────────────────────────────┐
│  Ollama Embedding API       │
│  model: nomic-embed-text    │
│                             │
│  POST /api/embeddings       │
│  {"model": "...",           │
│   "prompt": "..."}          │
└─────────────────────────────┘
     │
     ▼
   768-dim vector
     │
     ▼
┌─────────────────────────────┐
│  ChromaDB                   │
│  - HNSW index               │
│  - Cosine similarity        │
│  - Persistent storage       │
└─────────────────────────────┘
```

### Features

| Feature | Description |
|---------|-------------|
| Semantic | Understands meaning, not just keywords |
| Embeddings | nomic-embed-text via Ollama |
| Scalable | ChromaDB handles millions of vectors |
| Persistent | Survives container restarts |

---

## Hybrid Retriever

### Class: `HybridRetriever`

Unified interface combining both retrieval methods.

```python
from rag_uq.streaming_index import HybridRetriever

hybrid = HybridRetriever(
    bm25_persist_path="data/bm25_index",
    chroma_collection="rag_documents",
    chroma_persist_dir="./data/chroma_db",
    chroma_host="chromadb",
    chroma_port=8000
)

# Add documents to both indices
hybrid.add_documents(docs, batch_size=100)

# Retrieve with both scores
results = hybrid.retrieve(
    query="Python machine learning",
    top_k=10,
    alpha=0.5  # Weight for dense (ignored if using router)
)

# Get raw scores for router
bm25_scores, dense_scores, doc_ids, texts = hybrid.get_scores_for_router(
    query="Python machine learning",
    num_passages=20
)
```

### Fusion Methods

#### 1. Linear Combination (Simple)

```python
hybrid_score = alpha * dense_score + (1 - alpha) * bm25_score
```

#### 2. Learned Routing (Advanced)

```python
from rag_uq.router import RetrievalRouter

router = RetrievalRouter()
router.load_checkpoint("models/router_lora/best_router.pt")

# Get separate scores
bm25, dense, ids, texts = hybrid.get_scores_for_router(query, num_passages=20)

# Apply learned routing
weights = router(bm25, dense)
hybrid_scores = weights * dense + (1 - weights) * bm25

# Rerank
top_indices = hybrid_scores.argsort(descending=True)[:top_k]
```

---

## Streaming Index

### Class: `StreamingIndex`

For incremental document updates without full rebuild.

```python
from rag_uq.streaming_index import StreamingIndex

stream_idx = StreamingIndex(
    bm25_path="data/bm25_streaming",
    chroma_collection="streaming_docs"
)

# Stream documents one at a time
for doc in document_stream:
    stream_idx.add_document(doc)
    
# Or batch
stream_idx.add_documents(batch, batch_size=50)

# Periodic sync
stream_idx.sync()  # Ensures persistence
```

---

## Configuration Guide

### BM25 Tuning

| Parameter | Default | When to Adjust |
|-----------|---------|----------------|
| `k1` | 1.5 | Lower for short docs, higher for long |
| `b` | 0.75 | Lower if length variation matters less |

### ChromaDB Tuning

| Setting | Default | Notes |
|---------|---------|-------|
| Host | `chromadb` | Docker service name |
| Port | `8000` | Default ChromaDB port |
| Collection | `rag_documents` | Logical grouping |

### Embedding Model

| Model | Dimensions | Notes |
|-------|------------|-------|
| `nomic-embed-text` | 768 | Default, good balance |
| `mxbai-embed-large` | 1024 | Higher quality, slower |

---

## Best Practices

### 1. When to Use BM25

- ✅ Exact keyword matching needed
- ✅ Rare terms or proper nouns
- ✅ Structured queries
- ✅ Fast, low-resource retrieval

### 2. When to Use Dense

- ✅ Semantic similarity matters
- ✅ Paraphrase queries
- ✅ Cross-lingual (with appropriate model)
- ✅ Context understanding needed

### 3. When to Use Hybrid

- ✅ Unknown query distribution
- ✅ Best coverage across query types
- ✅ Production systems

### Index Size Estimation

| Documents | BM25 Size | ChromaDB Size |
|-----------|-----------|---------------|
| 10K | ~10 MB | ~50 MB |
| 100K | ~100 MB | ~500 MB |
| 1M | ~1 GB | ~5 GB |

---

## Example: Full Pipeline

```python
from rag_uq.streaming_index import HybridRetriever, Document
from rag_uq.router import RetrievalRouter
import torch

# 1. Initialize
hybrid = HybridRetriever(
    bm25_persist_path="data/bm25_index",
    chroma_collection="rag_documents"
)

router = RetrievalRouter()
router.load_checkpoint("models/router_lora/best_router.pt")
router.eval()

# 2. Index documents
docs = [Document(id=f"d{i}", text=text) for i, text in enumerate(corpus)]
hybrid.add_documents(docs)

# 3. Query with routing
query = "What is retrieval augmented generation?"

bm25_scores, dense_scores, doc_ids, texts = hybrid.get_scores_for_router(
    query, num_passages=20
)

with torch.no_grad():
    bm25_t = torch.tensor([bm25_scores])
    dense_t = torch.tensor([dense_scores])
    weights = router(bm25_t, dense_t)
    hybrid_scores = weights * dense_t + (1 - weights) * bm25_t

# 4. Get top results
top_k = 5
top_indices = hybrid_scores[0].argsort(descending=True)[:top_k]
results = [(doc_ids[i], texts[i], hybrid_scores[0][i].item()) 
           for i in top_indices]

for doc_id, text, score in results:
    print(f"{doc_id} ({score:.3f}): {text[:100]}...")
```
