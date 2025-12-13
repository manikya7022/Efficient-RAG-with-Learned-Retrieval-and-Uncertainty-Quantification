"""
Download required models for offline use.

Downloads:
- Sentence-transformers models (MiniLM-L6-v2)
- Any other required cached models

Run this once during Docker build or initial setup.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_sentence_transformers():
    """Download sentence-transformers models."""
    try:
        from sentence_transformers import SentenceTransformer
        
        models = [
            'all-MiniLM-L6-v2',  # ~90MB, general purpose
            # Add more models if needed
        ]
        
        for model_name in models:
            logger.info(f"Downloading {model_name}...")
            model = SentenceTransformer(model_name)
            logger.info(f"Successfully downloaded {model_name}")
            
    except ImportError:
        logger.warning("sentence-transformers not installed, skipping download")


def download_nltk_data():
    """Download NLTK data for text processing."""
    try:
        import nltk
        
        packages = ['punkt', 'stopwords', 'wordnet']
        
        for package in packages:
            try:
                nltk.download(package, quiet=True)
                logger.info(f"Downloaded NLTK {package}")
            except Exception as e:
                logger.warning(f"Failed to download NLTK {package}: {e}")
                
    except ImportError:
        logger.warning("NLTK not installed, skipping download")


def verify_downloads():
    """Verify all downloads are accessible."""
    logger.info("Verifying downloads...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(["test sentence"])
        logger.info(f"Sentence transformers OK (embedding dim: {len(embeddings[0])})")
    except Exception as e:
        logger.error(f"Sentence transformers failed: {e}")
    
    logger.info("Verification complete!")


if __name__ == "__main__":
    logger.info("Starting model downloads...")
    
    download_sentence_transformers()
    download_nltk_data()
    verify_downloads()
    
    logger.info("All downloads complete!")
