"""
Wikipedia Corpus Preparation

Downloads Wikipedia articles, chunks them into passages, and prepares
them for indexing. Designed for slow, resumable processing on local machines.

Key Features:
    - Streaming download to avoid memory issues
    - Rate-limited API calls
    - Resumable with checkpoints
    - Configurable chunking strategy
"""

import requests
import json
import time
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    chunk_size: int = 200  # Words per chunk
    overlap: int = 50      # Overlapping words
    min_chunk_length: int = 50  # Minimum characters
    max_chunk_length: int = 2000  # Maximum characters


def chunk_text(
    text: str, 
    config: Optional[ChunkConfig] = None
) -> List[str]:
    """
    Split text into overlapping chunks using sliding window.
    
    Args:
        text: Input text to chunk
        config: Chunking configuration
        
    Returns:
        List of text chunks
    """
    if config is None:
        config = ChunkConfig()
    
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    
    if len(words) < config.chunk_size // 2:
        return [text] if len(text) >= config.min_chunk_length else []
    
    chunks = []
    step = config.chunk_size - config.overlap
    
    for i in range(0, len(words), step):
        chunk_words = words[i:i + config.chunk_size]
        chunk = ' '.join(chunk_words)
        
        if len(chunk) >= config.min_chunk_length:
            # Truncate if too long
            if len(chunk) > config.max_chunk_length:
                chunk = chunk[:config.max_chunk_length]
            chunks.append(chunk)
        
        # Stop if we've covered all words
        if i + config.chunk_size >= len(words):
            break
    
    return chunks


class WikipediaDownloader:
    """
    Download Wikipedia articles via API.
    
    Uses the Wikipedia REST API to fetch article content.
    Implements rate limiting and checkpointing for reliability.
    
    Args:
        output_dir: Directory for output files
        rate_limit_delay: Seconds between API calls
        checkpoint_interval: Articles between checkpoints
    """
    
    API_BASE = "https://en.wikipedia.org/w/api.php"
    
    def __init__(
        self,
        output_dir: str = "data/raw",
        rate_limit_delay: float = 0.5,
        checkpoint_interval: int = 100
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.checkpoint_interval = checkpoint_interval
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAG-UQ-Research/1.0 (PhD Research Project)'
        })
    
    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make rate-limited API request."""
        time.sleep(self.rate_limit_delay)
        
        try:
            response = self.session.get(self.API_BASE, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {}
    
    def get_random_articles(self, n_articles: int = 100) -> List[str]:
        """Get titles of random articles."""
        titles = []
        
        while len(titles) < n_articles:
            params = {
                "action": "query",
                "list": "random",
                "rnnamespace": 0,  # Main namespace only
                "rnlimit": min(50, n_articles - len(titles)),
                "format": "json"
            }
            
            data = self._make_request(params)
            
            if 'query' in data and 'random' in data['query']:
                for item in data['query']['random']:
                    titles.append(item['title'])
            
            logger.info(f"Fetched {len(titles)}/{n_articles} article titles")
        
        return titles[:n_articles]
    
    def get_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        """Fetch full article content."""
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|info",
            "explaintext": True,
            "exsectionformat": "plain",
            "inprop": "url",
            "format": "json"
        }
        
        data = self._make_request(params)
        
        if 'query' not in data or 'pages' not in data['query']:
            return None
        
        pages = data['query']['pages']
        for page_id, page_data in pages.items():
            if page_id == '-1':
                continue
            
            return {
                'page_id': page_id,
                'title': page_data.get('title', title),
                'extract': page_data.get('extract', ''),
                'url': page_data.get('fullurl', f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")
            }
        
        return None
    
    def download_corpus(
        self,
        n_articles: int = 1000,
        output_file: str = "wikipedia_corpus.jsonl",
        resume: bool = True
    ) -> int:
        """
        Download and save Wikipedia articles.
        
        Args:
            n_articles: Number of articles to download
            output_file: Output JSONL filename
            resume: Whether to resume from checkpoint
            
        Returns:
            Number of articles downloaded
        """
        output_path = self.output_dir / output_file
        checkpoint_path = self.output_dir / f".{output_file}.checkpoint"
        
        # Load checkpoint
        downloaded_titles = set()
        if resume and checkpoint_path.exists():
            with open(checkpoint_path) as f:
                downloaded_titles = set(json.load(f))
            logger.info(f"Resuming from checkpoint with {len(downloaded_titles)} articles")
        
        # Get article titles
        titles = self.get_random_articles(n_articles + len(downloaded_titles))
        titles = [t for t in titles if t not in downloaded_titles][:n_articles]
        
        if not titles:
            logger.info("No new articles to download")
            return len(downloaded_titles)
        
        # Open output file in append mode
        mode = 'a' if resume and output_path.exists() else 'w'
        
        with open(output_path, mode) as f:
            for i, title in enumerate(titles):
                article = self.get_article_content(title)
                
                if article and article['extract']:
                    f.write(json.dumps(article) + '\n')
                    downloaded_titles.add(title)
                    
                    if (i + 1) % self.checkpoint_interval == 0:
                        with open(checkpoint_path, 'w') as cp:
                            json.dump(list(downloaded_titles), cp)
                        logger.info(f"Checkpoint saved: {len(downloaded_titles)} articles")
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Downloaded {i + 1}/{len(titles)} articles")
        
        # Final checkpoint
        with open(checkpoint_path, 'w') as cp:
            json.dump(list(downloaded_titles), cp)
        
        logger.info(f"Download complete: {len(downloaded_titles)} articles total")
        return len(downloaded_titles)


def prepare_passages(
    input_file: str,
    output_file: str,
    chunk_config: Optional[ChunkConfig] = None
) -> int:
    """
    Convert articles to chunked passages for indexing.
    
    Args:
        input_file: Path to input JSONL (articles)
        output_file: Path to output JSONL (passages)
        chunk_config: Chunking configuration
        
    Returns:
        Number of passages created
    """
    if chunk_config is None:
        chunk_config = ChunkConfig()
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_passages = 0
    
    with open(input_path) as fin, open(output_path, 'w') as fout:
        for line_num, line in enumerate(fin):
            try:
                article = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            
            # Chunk article text
            chunks = chunk_text(article.get('extract', ''), chunk_config)
            
            for i, chunk in enumerate(chunks):
                passage = {
                    'id': f"{article.get('page_id', line_num)}_{i}",
                    'text': chunk,
                    'title': article.get('title', ''),
                    'metadata': {
                        'source': 'wikipedia',
                        'url': article.get('url', ''),
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                }
                fout.write(json.dumps(passage) + '\n')
                total_passages += 1
            
            if (line_num + 1) % 100 == 0:
                logger.info(f"Processed {line_num + 1} articles, {total_passages} passages")
    
    logger.info(f"Created {total_passages} passages from {line_num + 1} articles")
    return total_passages


def prepare_natural_questions(
    output_dir: str = "data/preprocessed",
    n_samples: int = 3000
) -> int:
    """
    Download and prepare Natural Questions dataset.
    
    Uses HuggingFace datasets for convenient access.
    
    Args:
        output_dir: Output directory
        n_samples: Number of samples to include
        
    Returns:
        Number of samples prepared
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library required. Install with: pip install datasets")
        return 0
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load NQ dataset - use the simpler format
    logger.info("Loading Natural Questions dataset...")
    try:
        # Try loading the newer parquet-based format
        dataset = load_dataset("google-research-datasets/natural_questions", split="validation")
    except Exception as e:
        logger.warning(f"Failed to load standard NQ, trying web_questions: {e}")
        try:
            # Fallback to web_questions which has a simpler format
            dataset = load_dataset("web_questions", split="train")
        except Exception as e2:
            logger.error(f"Failed to load datasets: {e2}")
            # Create synthetic data for testing
            logger.info("Creating synthetic QA data for testing...")
            return create_synthetic_nq(output_path, n_samples)
    
    processed = []
    
    for i, example in enumerate(dataset):
        if i >= n_samples:
            break
        
        try:
            # Handle different dataset formats
            if 'question' in example:
                # Check if question is dict or string
                if isinstance(example['question'], dict):
                    question = example['question'].get('text', str(example['question']))
                else:
                    question = str(example['question'])
            else:
                continue
            
            # Extract answers - handle multiple formats
            short_answers = []
            
            # Try annotations format (original NQ)
            if 'annotations' in example:
                annotations = example.get('annotations', {})
                if isinstance(annotations, dict):
                    sa_list = annotations.get('short_answers', [])
                    if isinstance(sa_list, dict):
                        starts = sa_list.get('start_token', [])
                        ends = sa_list.get('end_token', [])
                        tokens = example.get('document', {}).get('tokens', {}).get('token', [])
                        if tokens and starts and ends:
                            for start, end in zip(starts[:3], ends[:3]):  # Limit to 3 answers
                                if start < len(tokens) and end <= len(tokens):
                                    answer_text = ' '.join(tokens[start:end])
                                    if answer_text.strip():
                                        short_answers.append(answer_text)
            
            # Try answers format (web_questions and others)
            if not short_answers and 'answers' in example:
                answers = example['answers']
                if isinstance(answers, list):
                    short_answers = [str(a) for a in answers[:3] if a]
                elif isinstance(answers, str):
                    short_answers = [answers]
            
            # Get document context
            doc_text = ""
            if 'document' in example:
                doc = example.get('document', {})
                if isinstance(doc, dict):
                    tokens = doc.get('tokens', {})
                    if isinstance(tokens, dict):
                        token_list = tokens.get('token', [])
                        if token_list:
                            doc_text = ' '.join(token_list)
            
            # If no context, use question as context (for simpler datasets)
            if not doc_text:
                doc_text = question
            
            if question and short_answers:
                processed.append({
                    'id': f"nq_{i}",
                    'question': question,
                    'answers': list(set(short_answers)),
                    'context': doc_text[:5000],  # Truncate long contexts
                    'metadata': {
                        'source': 'natural_questions',
                        'example_id': str(example.get('id', i))
                    }
                })
        except Exception as e:
            logger.warning(f"Error processing example {i}: {e}")
            continue
        
        if (i + 1) % 500 == 0:
            logger.info(f"Processed {i + 1}/{n_samples} NQ examples, kept {len(processed)}")
    
    # Save to JSONL
    output_file = output_path / "nq_dev_3000.jsonl"
    with open(output_file, 'w') as f:
        for item in processed:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Saved {len(processed)} NQ examples to {output_file}")
    return len(processed)


def create_synthetic_nq(output_path: Path, n_samples: int) -> int:
    """Create synthetic QA data for testing when real data unavailable."""
    import random
    
    templates = [
        ("What is the capital of {country}?", "{capital}", "The capital of {country} is {capital}."),
        ("Who wrote {book}?", "{author}", "{author} wrote {book} in {year}."),
        ("When was {event}?", "{year}", "{event} occurred in {year}."),
        ("What is {concept}?", "{definition}", "{concept} is {definition}."),
    ]
    
    data = [
        {"country": "France", "capital": "Paris"},
        {"country": "Germany", "capital": "Berlin"},
        {"country": "Japan", "capital": "Tokyo"},
        {"book": "1984", "author": "George Orwell", "year": "1949"},
        {"book": "Pride and Prejudice", "author": "Jane Austen", "year": "1813"},
        {"event": "World War II", "year": "1939-1945"},
        {"concept": "Machine Learning", "definition": "a type of artificial intelligence"},
        {"concept": "RAG", "definition": "Retrieval-Augmented Generation"},
    ]
    
    processed = []
    for i in range(min(n_samples, len(data) * 10)):
        template = random.choice(templates)
        item = random.choice(data)
        
        try:
            question = template[0].format(**item)
            answer = template[1].format(**item)
            context = template[2].format(**item)
            
            processed.append({
                'id': f"syn_{i}",
                'question': question,
                'answers': [answer],
                'context': context,
                'metadata': {'source': 'synthetic'}
            })
        except KeyError:
            continue
    
    output_file = output_path / "nq_dev_3000.jsonl"
    with open(output_file, 'w') as f:
        for item in processed:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Created {len(processed)} synthetic examples")
    return len(processed)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare corpus for RAG")
    parser.add_argument("--task", choices=["download", "chunk", "nq", "all"], 
                       default="all", help="Task to run")
    parser.add_argument("--n-articles", type=int, default=1000,
                       help="Number of Wikipedia articles")
    parser.add_argument("--n-nq", type=int, default=3000,
                       help="Number of NQ samples")
    parser.add_argument("--output-dir", default="data",
                       help="Output directory")
    
    args = parser.parse_args()
    
    if args.task in ["download", "all"]:
        logger.info("Downloading Wikipedia articles...")
        downloader = WikipediaDownloader(output_dir=f"{args.output_dir}/raw")
        downloader.download_corpus(n_articles=args.n_articles)
    
    if args.task in ["chunk", "all"]:
        logger.info("Chunking articles into passages...")
        prepare_passages(
            input_file=f"{args.output_dir}/raw/wikipedia_corpus.jsonl",
            output_file=f"{args.output_dir}/preprocessed/wikipedia_100k.jsonl"
        )
    
    if args.task in ["nq", "all"]:
        logger.info("Preparing Natural Questions dataset...")
        prepare_natural_questions(
            output_dir=f"{args.output_dir}/preprocessed",
            n_samples=args.n_nq
        )
    
    logger.info("Corpus preparation complete!")
