"""
Router Training Experiment

Trains the learned retrieval router on Natural Questions dataset.
Includes data loading, training loop, and evaluation.

Usage:
    python run_router_training.py --nq-path data/preprocessed/nq_dev_3000.jsonl
    
Training on CPU: ~2 hours for 3000 queries with 50 epochs.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_uq.router import RetrievalRouter, RouterConfig, RouterTrainer, create_pseudo_labels
from rag_uq.streaming_index import HybridRetriever, Document


def load_nq_dataset(
    nq_path: str,
    max_samples: int = 3000
) -> List[Dict[str, Any]]:
    """Load Natural Questions dataset."""
    samples = []
    
    with open(nq_path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            
            try:
                sample = json.loads(line.strip())
                if sample.get('question') and sample.get('answers'):
                    samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(samples)} NQ samples")
    return samples


def prepare_training_data(
    nq_samples: List[Dict[str, Any]],
    retriever: HybridRetriever,
    num_passages: int = 20
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare training data for router.
    
    For each NQ question:
    1. Retrieve passages using both methods
    2. Create pseudo-relevance labels based on answer overlap
    3. Return stacked tensors
    """
    all_bm25_scores = []
    all_dense_scores = []
    all_relevances = []
    
    for sample in tqdm(nq_samples, desc="Preparing training data"):
        question = sample['question']
        answers = sample['answers']
        primary_answer = answers[0] if answers else ""
        
        # Get scores from retriever
        bm25_scores, dense_scores, doc_ids, texts = retriever.get_scores_for_router(
            question, num_passages=num_passages
        )
        
        # Create pseudo-labels based on answer overlap
        relevances = []
        for text in texts:
            if not text:
                relevances.append(0.0)
                continue
            
            # Check if any answer appears in the passage
            text_lower = text.lower()
            max_overlap = 0.0
            
            for answer in answers:
                answer_lower = answer.lower()
                if answer_lower in text_lower:
                    max_overlap = 1.0
                    break
                else:
                    # Partial overlap
                    answer_tokens = set(answer_lower.split())
                    text_tokens = set(text_lower.split())
                    if answer_tokens:
                        overlap = len(answer_tokens & text_tokens) / len(answer_tokens)
                        max_overlap = max(max_overlap, overlap)
            
            relevances.append(max_overlap)
        
        all_bm25_scores.append(bm25_scores)
        all_dense_scores.append(dense_scores)
        all_relevances.append(relevances)
    
    return (
        torch.tensor(all_bm25_scores, dtype=torch.float32),
        torch.tensor(all_dense_scores, dtype=torch.float32),
        torch.tensor(all_relevances, dtype=torch.float32)
    )


def train_router(
    nq_path: str,
    retriever: HybridRetriever,
    output_dir: str = "models/router_lora",
    hidden_dim: int = 64,
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    val_split: float = 0.1
) -> Dict[str, Any]:
    """
    Full router training pipeline.
    
    Args:
        nq_path: Path to NQ dataset
        retriever: Initialized hybrid retriever
        output_dir: Directory for model checkpoints
        hidden_dim: Router hidden dimension
        num_epochs: Training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        val_split: Fraction for validation
        
    Returns:
        Training results dictionary
    """
    # Load data
    nq_samples = load_nq_dataset(nq_path)
    
    # Split train/val
    n_val = int(len(nq_samples) * val_split)
    val_samples = nq_samples[:n_val]
    train_samples = nq_samples[n_val:]
    
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Prepare training data
    logger.info("Preparing training data...")
    train_bm25, train_dense, train_rel = prepare_training_data(train_samples, retriever)
    val_bm25, val_dense, val_rel = prepare_training_data(val_samples, retriever)
    
    # Initialize router
    config = RouterConfig(hidden_dim=hidden_dim, dropout=0.1)
    router = RetrievalRouter(config)
    
    logger.info(f"Router parameters: {sum(p.numel() for p in router.parameters()):,}")
    
    # Initialize trainer
    trainer = RouterTrainer(
        router=router,
        learning_rate=learning_rate,
        checkpoint_dir=output_dir
    )
    
    # Train
    logger.info("Starting training...")
    history = trainer.fit(
        train_data=(train_bm25, train_dense, train_rel),
        val_data=(val_bm25, val_dense, val_rel),
        num_epochs=num_epochs,
        batch_size=batch_size,
        early_stopping_patience=10
    )
    
    # Save final model
    trainer.save_checkpoint(f"{output_dir}/final_router.pt")
    
    # Evaluate
    router.eval()
    with torch.no_grad():
        val_weights = router(val_bm25, val_dense, update_stats=False)
        val_hybrid = val_weights * val_dense + (1 - val_weights) * val_bm25
        
        # Compute retrieval metrics
        # Check if top-1 retrieval contains relevant passage
        top1_relevance = []
        for i in range(len(val_rel)):
            top_idx = val_hybrid[i].argmax().item()
            top1_relevance.append(float(val_rel[i, top_idx] > 0.5))
        
        hit_at_1 = np.mean(top1_relevance)
    
    results = {
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'final_train_loss': history['train_losses'][-1] if history['train_losses'] else None,
        'final_val_loss': history['val_losses'][-1] if history['val_losses'] else None,
        'best_val_loss': min(history['val_losses']) if history['val_losses'] else None,
        'epochs_trained': len(history['train_losses']),
        'hit_at_1': hit_at_1,
        'model_path': f"{output_dir}/final_router.pt"
    }
    
    logger.info(f"Training complete. Results: {results}")
    
    # Save training curves
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history['train_losses'], label='Train Loss')
        if history['val_losses']:
            ax.plot(history['val_losses'], label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Router Training Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(f"{output_dir}/training_curves.png", dpi=150)
        plt.close()
        
        results['training_curves_path'] = f"{output_dir}/training_curves.png"
    except ImportError:
        pass
    
    return results


def run_synthetic_experiment(
    output_dir: str = "models/router_lora"
) -> Dict[str, Any]:
    """
    Run training with synthetic data (for testing without retriever).
    
    Creates fake BM25/dense scores and labels for quick validation.
    """
    logger.info("Running synthetic experiment...")
    
    # Create synthetic data
    n_samples = 500
    n_passages = 20
    
    # Simulate scenarios where BM25 or dense is better
    np.random.seed(42)
    
    bm25_scores = np.random.uniform(0, 10, (n_samples, n_passages))
    dense_scores = np.random.uniform(0, 1, (n_samples, n_passages))
    
    # Create relevance labels favoring one method or the other
    relevances = np.zeros((n_samples, n_passages))
    for i in range(n_samples):
        if i % 2 == 0:
            # BM25 better: put relevant docs where BM25 is high
            top_bm25_idx = np.argmax(bm25_scores[i])
            relevances[i, top_bm25_idx] = 1.0
        else:
            # Dense better: put relevant docs where dense is high
            top_dense_idx = np.argmax(dense_scores[i])
            relevances[i, top_dense_idx] = 1.0
    
    # Convert to tensors
    train_bm25 = torch.tensor(bm25_scores[:400], dtype=torch.float32)
    train_dense = torch.tensor(dense_scores[:400], dtype=torch.float32)
    train_rel = torch.tensor(relevances[:400], dtype=torch.float32)
    
    val_bm25 = torch.tensor(bm25_scores[400:], dtype=torch.float32)
    val_dense = torch.tensor(dense_scores[400:], dtype=torch.float32)
    val_rel = torch.tensor(relevances[400:], dtype=torch.float32)
    
    # Train
    config = RouterConfig(hidden_dim=32, dropout=0.1)
    router = RetrievalRouter(config)
    
    trainer = RouterTrainer(
        router=router,
        learning_rate=1e-3,
        checkpoint_dir=output_dir
    )
    
    history = trainer.fit(
        train_data=(train_bm25, train_dense, train_rel),
        val_data=(val_bm25, val_dense, val_rel),
        num_epochs=20,
        batch_size=32
    )
    
    trainer.save_checkpoint(f"{output_dir}/synthetic_router.pt")
    
    return {
        'type': 'synthetic',
        'train_samples': 400,
        'val_samples': 100,
        'final_train_loss': history['train_losses'][-1],
        'final_val_loss': history['val_losses'][-1] if history['val_losses'] else None,
        'model_path': f"{output_dir}/synthetic_router.pt"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train retrieval router")
    parser.add_argument("--nq-path", default="data/preprocessed/nq_dev_3000.jsonl",
                       help="Path to NQ dataset")
    parser.add_argument("--chroma-path", default="./data/chroma_db",
                       help="ChromaDB path")
    parser.add_argument("--bm25-path", default="./data/bm25_index.pkl",
                       help="BM25 index path")
    parser.add_argument("--output-dir", default="models/router_lora",
                       help="Output directory for checkpoints")
    parser.add_argument("--hidden-dim", type=int, default=64,
                       help="Router hidden dimension")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--synthetic", action="store_true",
                       help="Run synthetic experiment only")
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.synthetic:
        results = run_synthetic_experiment(args.output_dir)
    else:
        # Initialize retriever
        retriever = HybridRetriever(
            bm25_persist_path=args.bm25_path,
            chroma_persist_path=args.chroma_path
        )
        
        if len(retriever) == 0:
            logger.warning("Retriever is empty! Running synthetic experiment instead.")
            results = run_synthetic_experiment(args.output_dir)
        else:
            results = train_router(
                nq_path=args.nq_path,
                retriever=retriever,
                output_dir=args.output_dir,
                hidden_dim=args.hidden_dim,
                num_epochs=args.epochs,
                batch_size=args.batch_size
            )
    
    # Save results
    with open(f"{args.output_dir}/training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining Results:")
    print(json.dumps(results, indent=2))
