"""
RAG with Learned Retrieval and Uncertainty Quantification

A PhD-level research implementation combining differentiable retrieval gating
with Bayesian uncertainty calibration for retrieval-augmented generation.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from rag_uq.router import RetrievalRouter, ApproxNDCGLoss
from rag_uq.confidence import MCDropoutConfidence, ConformalRAG
from rag_uq.streaming_index import HybridRetriever, StreamingIndex
from rag_uq.eval_protocol import RAGEvaluator

__all__ = [
    "RetrievalRouter",
    "ApproxNDCGLoss",
    "MCDropoutConfidence",
    "ConformalRAG",
    "HybridRetriever",
    "StreamingIndex",
    "RAGEvaluator",
]
