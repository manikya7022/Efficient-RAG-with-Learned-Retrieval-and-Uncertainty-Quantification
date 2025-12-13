"""
ChromaDB Index Builder

Incrementally builds the dense retrieval index from preprocessed passages.
Designed for memory-efficient processing on local machines.

Usage:
    python build_chroma_index.py --corpus data/preprocessed/wikipedia_100k.jsonl
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_uq.streaming_index import HybridRetriever, Document, StreamingIndex


def build_index_from_jsonl(
    corpus_path: str,
    chroma_persist_path: str = "./data/chroma_db",
    bm25_persist_path: str = "./data/bm25_index.pkl",
    batch_size: int = 100,
    resume: bool = True,
    chroma_host: Optional[str] = None
) -> int:
    """
    Build hybrid index from JSONL corpus.
    
    Args:
        corpus_path: Path to JSONL corpus file
        chroma_persist_path: ChromaDB persistence directory
        bm25_persist_path: BM25 index persistence path
        batch_size: Documents per indexing batch
        resume: Whether to resume from checkpoint
        chroma_host: Optional ChromaDB host for Docker
        
    Returns:
        Total number of documents indexed
    """
    # Ensure paths exist
    Path(chroma_persist_path).parent.mkdir(parents=True, exist_ok=True)
    Path(bm25_persist_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize retriever
    retriever = HybridRetriever(
        bm25_persist_path=bm25_persist_path,
        chroma_persist_path=chroma_persist_path,
        chroma_host=chroma_host or os.environ.get('CHROMA_HOST')
    )
    
    # Use streaming indexer
    streaming = StreamingIndex(
        retriever=retriever,
        checkpoint_path=f"{chroma_persist_path}/../index_checkpoint.json",
        batch_size=batch_size
    )
    
    # Stream and index
    total = 0
    for batch_count in streaming.stream_from_jsonl(corpus_path, resume=resume):
        total += batch_count
    
    progress = streaming.get_progress()
    logger.info(f"Indexing complete. Total documents: {progress['total_indexed']}")
    
    return progress['total_indexed']


def verify_index(
    chroma_persist_path: str = "./data/chroma_db",
    bm25_persist_path: str = "./data/bm25_index.pkl",
    test_queries: Optional[list] = None
) -> dict:
    """
    Verify index integrity and test retrieval.
    
    Args:
        chroma_persist_path: ChromaDB persistence directory
        bm25_persist_path: BM25 index persistence path
        test_queries: Optional list of test queries
        
    Returns:
        Verification results
    """
    retriever = HybridRetriever(
        bm25_persist_path=bm25_persist_path,
        chroma_persist_path=chroma_persist_path
    )
    
    if test_queries is None:
        test_queries = [
            "What is machine learning?",
            "Who invented the telephone?",
            "What is the capital of France?"
        ]
    
    results = {
        'total_documents': len(retriever),
        'bm25_size': len(retriever.bm25_index) if retriever.bm25_index else 0,
        'dense_size': len(retriever.dense_index) if retriever.dense_index else 0,
        'test_results': []
    }
    
    for query in test_queries:
        search_results = retriever.hybrid_search(query, top_k=3)
        
        results['test_results'].append({
            'query': query,
            'num_results': len(search_results),
            'top_result': {
                'text': search_results[0].text[:200] + '...' if search_results else '',
                'bm25_score': search_results[0].bm25_score if search_results else 0,
                'dense_score': search_results[0].dense_score if search_results else 0
            } if search_results else None
        })
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB index")
    parser.add_argument("--corpus", default="data/preprocessed/wikipedia_100k.jsonl",
                       help="Path to corpus JSONL")
    parser.add_argument("--chroma-path", default="./data/chroma_db",
                       help="ChromaDB persistence path")
    parser.add_argument("--bm25-path", default="./data/bm25_index.pkl",
                       help="BM25 index path")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for indexing")
    parser.add_argument("--no-resume", action="store_true",
                       help="Don't resume from checkpoint")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing index")
    parser.add_argument("--chroma-host", default=None,
                       help="ChromaDB host for Docker")
    
    args = parser.parse_args()
    
    if args.verify_only:
        results = verify_index(
            chroma_persist_path=args.chroma_path,
            bm25_persist_path=args.bm25_path
        )
        print(json.dumps(results, indent=2))
    else:
        total = build_index_from_jsonl(
            corpus_path=args.corpus,
            chroma_persist_path=args.chroma_path,
            bm25_persist_path=args.bm25_path,
            batch_size=args.batch_size,
            resume=not args.no_resume,
            chroma_host=args.chroma_host
        )
        
        # Verify after building
        results = verify_index(
            chroma_persist_path=args.chroma_path,
            bm25_persist_path=args.bm25_path
        )
        print("\nVerification Results:")
        print(json.dumps(results, indent=2))
