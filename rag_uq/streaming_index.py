"""
Hybrid Retrieval with Streaming Index

Implements a unified retrieval interface combining:
1. Dense retrieval via ChromaDB with Ollama embeddings
2. Sparse retrieval via BM25 (rank-bm25)

Supports incremental document addition without full re-indexing,
making it suitable for streaming/online scenarios.

Key Features:
    - Unified query interface returning both BM25 and dense scores
    - Batch-based incremental indexing for memory efficiency
    - Persistence support for both indices
    - Automatic synchronization between dense and sparse indices
"""

from typing import List, Dict, Any, Optional, Tuple, Iterator
import json
import pickle
import logging
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import hashlib
import os

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    logger.warning("chromadb not installed. Dense retrieval disabled.")

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger.warning("rank-bm25 not installed. Sparse retrieval disabled.")

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    logger.warning("ollama not installed. Using fallback embeddings.")


@dataclass
class Document:
    """A document for indexing."""
    id: str
    text: str
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'text': self.text,
            'title': self.title or '',
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        return cls(
            id=data['id'],
            text=data['text'],
            title=data.get('title'),
            metadata=data.get('metadata')
        )


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""
    doc_id: str
    text: str
    bm25_score: float
    dense_score: float
    hybrid_score: Optional[float] = None
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BM25Index:
    """
    BM25 sparse retrieval index with persistence support.
    
    Uses BM25Okapi for scoring with configurable tokenization.
    Supports incremental document addition.
    """
    
    def __init__(
        self,
        persist_path: Optional[str] = None,
        k1: float = 1.5,
        b: float = 0.75
    ):
        self.persist_path = Path(persist_path) if persist_path else None
        self.k1 = k1
        self.b = b
        
        self.documents: Dict[str, Document] = {}
        self.doc_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        
        if self.persist_path and self.persist_path.exists():
            self._load()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the index.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Number of new documents added
        """
        new_count = 0
        for doc in documents:
            if doc.id not in self.documents:
                self.documents[doc.id] = doc
                self.doc_ids.append(doc.id)
                self.tokenized_corpus.append(self._tokenize(doc.text))
                new_count += 1
        
        # Rebuild BM25 index
        if new_count > 0 and self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
            logger.info(f"Added {new_count} documents to BM25 index. Total: {len(self.doc_ids)}")
        
        if self.persist_path:
            self._save()
        
        return new_count
    
    def search(
        self, 
        query: str, 
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search the index.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, score) tuples
        """
        if self.bm25 is None or not self.doc_ids:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.doc_ids[idx], float(scores[idx])))
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve document by ID."""
        return self.documents.get(doc_id)
    
    def _save(self):
        """Persist index to disk."""
        if self.persist_path is None:
            return
        
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'documents': {k: v.to_dict() for k, v in self.documents.items()},
            'doc_ids': self.doc_ids,
            'tokenized_corpus': self.tokenized_corpus,
            'k1': self.k1,
            'b': self.b
        }
        
        with open(self.persist_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.debug(f"Saved BM25 index to {self.persist_path}")
    
    def _load(self):
        """Load index from disk."""
        if self.persist_path is None or not self.persist_path.exists():
            return
        
        with open(self.persist_path, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = {k: Document.from_dict(v) for k, v in data['documents'].items()}
        self.doc_ids = data['doc_ids']
        self.tokenized_corpus = data['tokenized_corpus']
        self.k1 = data['k1']
        self.b = data['b']
        
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        
        logger.info(f"Loaded BM25 index with {len(self.doc_ids)} documents")
    
    def __len__(self) -> int:
        return len(self.doc_ids)


class DenseIndex:
    """
    Dense retrieval index using ChromaDB with Ollama embeddings.
    
    Supports batch-based incremental indexing for memory efficiency.
    Uses nomic-embed-text model for embeddings via Ollama.
    """
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./data/chroma_db",
        embedding_model: str = "nomic-embed-text",
        chroma_host: Optional[str] = None,
        chroma_port: int = 8000
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        if not HAS_CHROMA:
            raise ImportError("chromadb is required for DenseIndex")
        
        # Connect to ChromaDB
        if chroma_host:
            # Use HTTP client for Docker deployment
            self.client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        else:
            # Use persistent local client
            self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized DenseIndex with collection '{collection_name}'")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Ollama."""
        if not HAS_OLLAMA:
            # Fallback: simple hash-based pseudo-embedding (for testing only)
            import hashlib
            hash_bytes = hashlib.sha256(text.encode()).digest()
            return [float(b) / 255.0 for b in hash_bytes][:384]
        
        try:
            response = ollama.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            # Return zero vector on failure
            return [0.0] * 768
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        return [self._get_embedding(text) for text in texts]
    
    def add_documents(
        self, 
        documents: List[Document],
        batch_size: int = 100
    ) -> int:
        """
        Add documents to the index in batches.
        
        Args:
            documents: List of Document objects
            batch_size: Documents per batch for memory efficiency
            
        Returns:
            Number of documents added
        """
        # Filter out already indexed documents
        existing_ids = set(self.collection.get()['ids'])
        new_docs = [d for d in documents if d.id not in existing_ids]
        
        if not new_docs:
            logger.info("No new documents to add")
            return 0
        
        # Process in batches
        total_added = 0
        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i:i + batch_size]
            
            ids = [d.id for d in batch]
            texts = [d.text for d in batch]
            metadatas = [
                {'title': d.title or '', **(d.metadata or {})}
                for d in batch
            ]
            embeddings = self._get_embeddings_batch(texts)
            
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            total_added += len(batch)
            logger.info(f"Indexed batch {i//batch_size + 1}, total: {total_added}/{len(new_docs)}")
        
        return total_added
    
    def search(
        self, 
        query: str, 
        top_k: int = 10
    ) -> List[Tuple[str, float, str]]:
        """
        Search the index.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (doc_id, score, text) tuples
        """
        query_embedding = self._get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'distances', 'metadatas']
        )
        
        # Convert distances to similarity scores
        # ChromaDB returns L2 distance for cosine space: distance = 1 - cosine_sim
        output = []
        for i, doc_id in enumerate(results['ids'][0]):
            distance = results['distances'][0][i]
            score = 1.0 - distance  # Convert to similarity
            text = results['documents'][0][i]
            output.append((doc_id, float(score), text))
        
        return output
    
    def __len__(self) -> int:
        return self.collection.count()


class HybridRetriever:
    """
    Unified hybrid retrieval combining BM25 and dense retrieval.
    
    Returns both sets of scores for use with the learned router.
    Supports incremental updates to both indices.
    
    Args:
        bm25_persist_path: Path for BM25 index persistence
        chroma_persist_path: Path for ChromaDB persistence
        chroma_host: Optional ChromaDB host for Docker deployment
        embedding_model: Ollama embedding model name
        
    Example:
        >>> retriever = HybridRetriever()
        >>> retriever.add_documents([Document(id="1", text="The sky is blue.")])
        >>> results = retriever.hybrid_search("What color is the sky?", top_k=5)
        >>> for r in results:
        ...     print(f"{r.doc_id}: BM25={r.bm25_score:.3f}, Dense={r.dense_score:.3f}")
    """
    
    def __init__(
        self,
        bm25_persist_path: str = "./data/bm25_index.pkl",
        chroma_persist_path: str = "./data/chroma_db",
        chroma_host: Optional[str] = None,
        embedding_model: str = "nomic-embed-text"
    ):
        # Initialize BM25 index
        if HAS_BM25:
            self.bm25_index = BM25Index(persist_path=bm25_persist_path)
        else:
            self.bm25_index = None
            logger.warning("BM25 retrieval disabled")
        
        # Initialize dense index
        if HAS_CHROMA:
            self.dense_index = DenseIndex(
                persist_directory=chroma_persist_path,
                chroma_host=chroma_host or os.environ.get('CHROMA_HOST'),
                embedding_model=embedding_model
            )
        else:
            self.dense_index = None
            logger.warning("Dense retrieval disabled")
        
        # Document store for text retrieval
        self.documents: Dict[str, Document] = {}
    
    def add_documents(
        self, 
        documents: List[Document],
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Add documents to both indices.
        
        Returns:
            Dictionary with counts from each index
        """
        # Store documents
        for doc in documents:
            self.documents[doc.id] = doc
        
        stats = {}
        
        if self.bm25_index:
            stats['bm25_added'] = self.bm25_index.add_documents(documents)
        
        if self.dense_index:
            stats['dense_added'] = self.dense_index.add_documents(documents, batch_size)
        
        stats['total_documents'] = len(self.documents)
        return stats
    
    def bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Search using BM25 only."""
        if self.bm25_index is None:
            return []
        return self.bm25_index.search(query, top_k)
    
    def dense_search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Search using dense retrieval only."""
        if self.dense_index is None:
            return []
        results = self.dense_index.search(query, top_k)
        return [(doc_id, score) for doc_id, score, _ in results]
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        retrieval_pool_size: int = 50
    ) -> List[RetrievalResult]:
        """
        Perform hybrid search returning both BM25 and dense scores.
        
        First retrieves a larger pool from each method, then merges and
        returns top_k results with scores from both.
        
        Args:
            query: Search query
            top_k: Final number of results
            retrieval_pool_size: Pool size from each retriever
            
        Returns:
            List of RetrievalResult with both scores
        """
        # Get results from both retrievers
        bm25_results = dict(self.bm25_search(query, retrieval_pool_size))
        dense_results = dict(self.dense_search(query, retrieval_pool_size))
        
        # Get union of all doc IDs
        all_doc_ids = set(bm25_results.keys()) | set(dense_results.keys())
        
        # Build combined results
        results = []
        for doc_id in all_doc_ids:
            doc = self.documents.get(doc_id)
            if doc is None:
                continue
            
            bm25_score = bm25_results.get(doc_id, 0.0)
            dense_score = dense_results.get(doc_id, 0.0)
            
            results.append(RetrievalResult(
                doc_id=doc_id,
                text=doc.text,
                bm25_score=bm25_score,
                dense_score=dense_score,
                title=doc.title,
                metadata=doc.metadata
            ))
        
        # Sort by sum of normalized scores (simple fusion)
        # Normalize scores to [0, 1]
        if results:
            max_bm25 = max(r.bm25_score for r in results) or 1
            max_dense = max(r.dense_score for r in results) or 1
            
            for r in results:
                r.hybrid_score = (
                    r.bm25_score / max_bm25 + r.dense_score / max_dense
                ) / 2
            
            results.sort(key=lambda x: x.hybrid_score or 0, reverse=True)
        
        return results[:top_k]
    
    def get_scores_for_router(
        self,
        query: str,
        num_passages: int = 20
    ) -> Tuple[List[float], List[float], List[str], List[str]]:
        """
        Get parallel score arrays for the router model.
        
        Returns:
            Tuple of (bm25_scores, dense_scores, doc_ids, texts)
            All lists are aligned and of length num_passages
        """
        results = self.hybrid_search(query, top_k=num_passages)
        
        bm25_scores = []
        dense_scores = []
        doc_ids = []
        texts = []
        
        for r in results:
            bm25_scores.append(r.bm25_score)
            dense_scores.append(r.dense_score)
            doc_ids.append(r.doc_id)
            texts.append(r.text)
        
        # Pad if needed
        while len(bm25_scores) < num_passages:
            bm25_scores.append(0.0)
            dense_scores.append(0.0)
            doc_ids.append("")
            texts.append("")
        
        return bm25_scores, dense_scores, doc_ids, texts
    
    def __len__(self) -> int:
        return len(self.documents)


class StreamingIndex:
    """
    Streaming document indexer for incremental corpus building.
    
    Designed for slow, resumable indexing of large corpora on local machines.
    Tracks progress and supports checkpoint/resume.
    
    Args:
        retriever: HybridRetriever instance
        checkpoint_path: Path for progress checkpoints
        batch_size: Documents per indexing batch
        
    Example:
        >>> index = StreamingIndex(retriever)
        >>> for batch in index.stream_from_jsonl("data/corpus.jsonl"):
        ...     print(f"Indexed {batch} documents")
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        checkpoint_path: str = "./data/index_checkpoint.json",
        batch_size: int = 100
    ):
        self.retriever = retriever
        self.checkpoint_path = Path(checkpoint_path)
        self.batch_size = batch_size
        
        self.progress = self._load_checkpoint()
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load indexing progress."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                return json.load(f)
        return {'last_offset': 0, 'total_indexed': 0, 'files_completed': []}
    
    def _save_checkpoint(self):
        """Save indexing progress."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.progress, f)
    
    def stream_from_jsonl(
        self,
        jsonl_path: str,
        resume: bool = True
    ) -> Iterator[int]:
        """
        Stream and index documents from a JSONL file.
        
        Each line should be a JSON object with 'id' and 'text' fields.
        Optionally 'title' and 'metadata'.
        
        Args:
            jsonl_path: Path to JSONL file
            resume: Whether to resume from checkpoint
            
        Yields:
            Number of documents indexed in each batch
        """
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {jsonl_path}")
        
        start_offset = self.progress['last_offset'] if resume else 0
        
        with open(path) as f:
            # Skip to checkpoint
            for _ in range(start_offset):
                f.readline()
            
            batch = []
            offset = start_offset
            
            for line in f:
                try:
                    data = json.loads(line.strip())
                    doc = Document(
                        id=data['id'],
                        text=data['text'],
                        title=data.get('title'),
                        metadata=data.get('metadata')
                    )
                    batch.append(doc)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping invalid line at offset {offset}: {e}")
                
                offset += 1
                
                if len(batch) >= self.batch_size:
                    stats = self.retriever.add_documents(batch)
                    self.progress['last_offset'] = offset
                    self.progress['total_indexed'] += len(batch)
                    self._save_checkpoint()
                    
                    logger.info(
                        f"Indexed batch: {len(batch)} docs, "
                        f"total: {self.progress['total_indexed']}"
                    )
                    yield len(batch)
                    batch = []
            
            # Final batch
            if batch:
                self.retriever.add_documents(batch)
                self.progress['last_offset'] = offset
                self.progress['total_indexed'] += len(batch)
                self._save_checkpoint()
                yield len(batch)
        
        # Mark file as completed
        if jsonl_path not in self.progress['files_completed']:
            self.progress['files_completed'].append(jsonl_path)
            self._save_checkpoint()
        
        logger.info(f"Completed indexing {jsonl_path}")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get indexing progress."""
        return {
            **self.progress,
            'retriever_size': len(self.retriever)
        }
