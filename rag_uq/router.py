"""
Learned Retrieval Router

A differentiable gating network that learns to weight BM25 vs. dense retrieval
scores on a per-passage basis. This implements the core novelty of the system:
dynamic retrieval strategy selection based on query-passage characteristics.

Architecture:
    - Lightweight MLP (~0.5M parameters)
    - Input: [BM25_score, dense_score, normalized_score_diff]
    - Output: Per-passage gating weight (0=BM25 only, 1=dense only)

Training:
    - Uses ApproxNDCG listwise ranking loss
    - Trainable on CPU in ~2 hours (batch_size=16, 3K queries)
    - Freeze LLM and retrievers; only train router

Reference:
    - ApproxNDCG: "A General Approximation Framework for Direct Optimization of
      Information Retrieval Measures" (Qin et al., 2010)
"""

from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RouterConfig:
    """Configuration for the RetrievalRouter."""
    hidden_dim: int = 64
    dropout: float = 0.1
    temperature: float = 1.0  # For ApproxNDCG
    num_layers: int = 2
    use_batch_norm: bool = False


class RetrievalRouter(nn.Module):
    """
    Learned retrieval selector that combines BM25 and dense retrieval scores.
    
    The router takes normalized scores from both retrievers and outputs a 
    per-passage gating weight that determines the contribution of each 
    retrieval method to the final ranking.
    
    Args:
        config: RouterConfig with architecture hyperparameters
        
    Example:
        >>> router = RetrievalRouter(RouterConfig(hidden_dim=64))
        >>> bm25_scores = torch.randn(4, 20)  # batch_size=4, num_passages=20
        >>> dense_scores = torch.randn(4, 20)
        >>> weights = router(bm25_scores, dense_scores)
        >>> print(weights.shape)  # torch.Size([4, 20])
    """
    
    def __init__(self, config: Optional[RouterConfig] = None):
        super().__init__()
        self.config = config or RouterConfig()
        
        # Input: [BM25_score, dense_score, normalized_score_diff]
        input_dim = 3
        
        layers = []
        current_dim = input_dim
        
        for i in range(self.config.num_layers - 1):
            layers.append(nn.Linear(current_dim, self.config.hidden_dim))
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(self.config.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config.dropout))
            current_dim = self.config.hidden_dim
        
        # Final layer outputs scalar gating weight
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.scorer = nn.Sequential(*layers)
        
        # Track statistics for normalization
        self.register_buffer('bm25_mean', torch.tensor(0.0))
        self.register_buffer('bm25_std', torch.tensor(1.0))
        self.register_buffer('dense_mean', torch.tensor(0.0))
        self.register_buffer('dense_std', torch.tensor(1.0))
        self.stats_initialized = False
        
        logger.info(f"Initialized RetrievalRouter with {self._count_params()} parameters")
    
    def _count_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _normalize_scores(
        self, 
        bm25_scores: torch.Tensor, 
        dense_scores: torch.Tensor,
        update_stats: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize scores to have zero mean and unit variance.
        
        Uses running statistics during training for stable normalization
        during inference.
        """
        eps = 1e-6
        
        if update_stats and self.training:
            # Update running statistics
            with torch.no_grad():
                batch_bm25_mean = bm25_scores.mean()
                batch_bm25_std = bm25_scores.std() + eps
                batch_dense_mean = dense_scores.mean()
                batch_dense_std = dense_scores.std() + eps
                
                # Exponential moving average
                momentum = 0.1
                self.bm25_mean = (1 - momentum) * self.bm25_mean + momentum * batch_bm25_mean
                self.bm25_std = (1 - momentum) * self.bm25_std + momentum * batch_bm25_std
                self.dense_mean = (1 - momentum) * self.dense_mean + momentum * batch_dense_mean
                self.dense_std = (1 - momentum) * self.dense_std + momentum * batch_dense_std
                self.stats_initialized = True
        
        if self.stats_initialized:
            bm25_norm = (bm25_scores - self.bm25_mean) / (self.bm25_std + eps)
            dense_norm = (dense_scores - self.dense_mean) / (self.dense_std + eps)
        else:
            # Batch-wise normalization if no stats yet
            bm25_norm = (bm25_scores - bm25_scores.mean()) / (bm25_scores.std() + eps)
            dense_norm = (dense_scores - dense_scores.mean()) / (dense_scores.std() + eps)
        
        return bm25_norm, dense_norm
    
    def forward(
        self, 
        bm25_scores: torch.Tensor, 
        dense_scores: torch.Tensor,
        update_stats: bool = True
    ) -> torch.Tensor:
        """
        Compute per-passage gating weights.
        
        Args:
            bm25_scores: Tensor of shape [batch_size, num_passages] with BM25 scores
            dense_scores: Tensor of shape [batch_size, num_passages] with dense scores
            update_stats: Whether to update running statistics during training
            
        Returns:
            gating_weights: Tensor of shape [batch_size, num_passages]
                           Values near 0 favor BM25, values near 1 favor dense retrieval
        """
        # Normalize scores
        bm25_norm, dense_norm = self._normalize_scores(
            bm25_scores, dense_scores, update_stats=update_stats
        )
        
        # Compute score difference as additional feature
        score_diff = dense_norm - bm25_norm
        
        # Stack features: [batch, passages, 3]
        features = torch.stack([bm25_norm, dense_norm, score_diff], dim=-1)
        
        # Flatten for MLP processing
        batch_size, num_passages, _ = features.shape
        features_flat = features.view(-1, 3)
        
        # Get gating weights
        gating_weights = self.scorer(features_flat)
        gating_weights = gating_weights.view(batch_size, num_passages)
        
        return gating_weights
    
    def hybrid_rerank(
        self,
        bm25_scores: torch.Tensor,
        dense_scores: torch.Tensor,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine retrieval scores using learned weights and return top-k results.
        
        Args:
            bm25_scores: BM25 retrieval scores [batch_size, num_passages]
            dense_scores: Dense retrieval scores [batch_size, num_passages]
            top_k: Number of top passages to return
            
        Returns:
            Tuple of (top_k_scores, top_k_indices)
        """
        weights = self.forward(bm25_scores, dense_scores, update_stats=False)
        
        # Weighted combination
        hybrid_scores = weights * dense_scores + (1 - weights) * bm25_scores
        
        # Return top-k
        return torch.topk(hybrid_scores, k=min(top_k, hybrid_scores.size(-1)), dim=-1)
    
    def get_routing_decision(
        self,
        bm25_scores: torch.Tensor,
        dense_scores: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Get interpretable routing decision for analysis.
        
        Returns:
            Dictionary with routing statistics and decisions
        """
        with torch.no_grad():
            weights = self.forward(bm25_scores, dense_scores, update_stats=False)
            
            avg_weight = weights.mean().item()
            std_weight = weights.std().item()
            
            # Classify passages
            dense_preferred = (weights > threshold).float().mean().item()
            bm25_preferred = (weights <= threshold).float().mean().item()
            
            return {
                'avg_dense_weight': avg_weight,
                'weight_std': std_weight,
                'dense_preferred_ratio': dense_preferred,
                'bm25_preferred_ratio': bm25_preferred,
                'routing_weights': weights.cpu().numpy()
            }


class ApproxNDCGLoss(nn.Module):
    """
    Approximate NDCG loss for listwise learning-to-rank.
    
    This loss function enables end-to-end training of the router by providing
    a differentiable approximation to the NDCG metric.
    
    Reference:
        Qin et al., "A General Approximation Framework for Direct Optimization
        of Information Retrieval Measures" (TOIS 2010)
    
    Args:
        temperature: Temperature for score-to-probability conversion.
                    Lower values make the approximation sharper.
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        predicted_scores: torch.Tensor,
        relevance_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute ApproxNDCG loss.
        
        Args:
            predicted_scores: Model predictions [batch_size, num_items]
            relevance_labels: Ground truth relevance [batch_size, num_items]
                            Values should be >= 0, typically binary or graded
            mask: Optional mask for padded items [batch_size, num_items]
            
        Returns:
            Scalar loss (negative ApproxNDCG, to be minimized)
        """
        if mask is not None:
            # Apply mask by setting masked scores to very negative
            predicted_scores = predicted_scores.masked_fill(~mask, float('-inf'))
            relevance_labels = relevance_labels.masked_fill(~mask, 0)
        
        # Compute approximate ranks using softmax
        # Higher scores get lower (better) approximate ranks
        approx_ranks = self._approx_ranks(predicted_scores)
        
        # Compute NDCG components
        dcg = self._dcg(relevance_labels, approx_ranks)
        
        # Ideal DCG (sorted relevances)
        sorted_relevances, _ = torch.sort(relevance_labels, descending=True, dim=-1)
        ideal_ranks = torch.arange(1, relevance_labels.size(-1) + 1, 
                                   device=relevance_labels.device, 
                                   dtype=relevance_labels.dtype)
        ideal_ranks = ideal_ranks.unsqueeze(0).expand_as(sorted_relevances)
        idcg = self._dcg(sorted_relevances, ideal_ranks)
        
        # NDCG
        ndcg = dcg / (idcg + 1e-10)
        
        # Return negative mean NDCG as loss
        return -ndcg.mean()
    
    def _approx_ranks(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute approximate ranks using softmax.
        
        For each item, its approximate rank is 1 + sum of probabilities 
        that other items have higher scores.
        """
        batch_size, num_items = scores.shape
        
        # Pairwise score differences
        scores_diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)
        
        # Soft comparison: probability that j > i
        probs = torch.sigmoid(scores_diff / self.temperature)
        
        # Approximate rank: 1 + sum of probabilities that other items rank higher
        approx_ranks = 1 + probs.sum(dim=-1) - 0.5  # -0.5 removes self-comparison
        
        return approx_ranks
    
    def _dcg(
        self, 
        relevances: torch.Tensor, 
        ranks: torch.Tensor
    ) -> torch.Tensor:
        """Compute DCG given relevances and ranks."""
        # DCG = sum((2^rel - 1) / log2(1 + rank))
        gains = (2 ** relevances) - 1
        discounts = torch.log2(1 + ranks)
        dcg = (gains / discounts).sum(dim=-1)
        return dcg


class RouterTrainer:
    """
    Training loop for the RetrievalRouter.
    
    Handles data loading, optimization, and checkpoint saving for
    CPU-based training.
    
    Args:
        router: RetrievalRouter model
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        checkpoint_dir: Directory for saving checkpoints
    """
    
    def __init__(
        self,
        router: RetrievalRouter,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        checkpoint_dir: str = "models/router_lora"
    ):
        self.router = router
        self.loss_fn = ApproxNDCGLoss(temperature=router.config.temperature)
        self.optimizer = torch.optim.AdamW(
            router.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        self.checkpoint_dir = checkpoint_dir
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(
        self,
        train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_data: Tuple of (bm25_scores, dense_scores, relevance_labels)
                       Each has shape [num_queries, num_passages]
                       
        Returns:
            Average training loss
        """
        self.router.train()
        bm25_scores, dense_scores, relevances = train_data
        
        # Forward pass
        self.optimizer.zero_grad()
        gating_weights = self.router(bm25_scores, dense_scores)
        
        # Compute hybrid scores
        hybrid_scores = gating_weights * dense_scores + (1 - gating_weights) * bm25_scores
        
        # Compute loss
        loss = self.loss_fn(hybrid_scores, relevances)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.router.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def validate(
        self,
        val_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> float:
        """Validate on held-out data."""
        self.router.eval()
        bm25_scores, dense_scores, relevances = val_data
        
        with torch.no_grad():
            gating_weights = self.router(bm25_scores, dense_scores, update_stats=False)
            hybrid_scores = gating_weights * dense_scores + (1 - gating_weights) * bm25_scores
            loss = self.loss_fn(hybrid_scores, relevances)
        
        return loss.item()
    
    def fit(
        self,
        train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        val_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        num_epochs: int = 50,
        batch_size: int = 16,
        early_stopping_patience: int = 10
    ) -> Dict[str, list]:
        """
        Full training loop with early stopping.
        
        Args:
            train_data: Training data tuple
            val_data: Optional validation data tuple
            num_epochs: Maximum number of epochs
            batch_size: Batch size for mini-batch training
            early_stopping_patience: Epochs to wait for improvement
            
        Returns:
            Dictionary with training history
        """
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        bm25_train, dense_train, rel_train = train_data
        num_samples = bm25_train.size(0)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Shuffle data
            perm = torch.randperm(num_samples)
            bm25_shuffled = bm25_train[perm]
            dense_shuffled = dense_train[perm]
            rel_shuffled = rel_train[perm]
            
            # Mini-batch training
            epoch_losses = []
            for i in range(0, num_samples, batch_size):
                batch_bm25 = bm25_shuffled[i:i+batch_size]
                batch_dense = dense_shuffled[i:i+batch_size]
                batch_rel = rel_shuffled[i:i+batch_size]
                
                loss = self.train_epoch((batch_bm25, batch_dense, batch_rel))
                epoch_losses.append(loss)
            
            avg_train_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_train_loss)
            
            # Validation
            if val_data is not None:
                val_loss = self.validate(val_data)
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint(os.path.join(self.checkpoint_dir, 'best_router.pt'))
                else:
                    patience_counter += 1
                
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.router.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.router.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        self.router.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Loaded checkpoint from {path}")


def create_pseudo_labels(
    bm25_passages: list,
    dense_passages: list,
    answer: str,
    num_passages: int = 20
) -> torch.Tensor:
    """
    Create pseudo-relevance labels based on answer overlap.
    
    For training without human annotations, we use the presence of
    the answer string in each passage as a proxy for relevance.
    
    Args:
        bm25_passages: List of passage texts from BM25
        dense_passages: List of passage texts from dense retrieval
        answer: Ground truth answer string
        num_passages: Number of passages to score
        
    Returns:
        Relevance labels tensor [num_passages]
    """
    labels = []
    answer_lower = answer.lower()
    
    # Combine passages (assuming same ordering)
    all_passages = set(bm25_passages[:num_passages] + dense_passages[:num_passages])
    
    for passage in list(all_passages)[:num_passages]:
        if answer_lower in passage.lower():
            labels.append(1.0)
        else:
            # Partial match: check for significant overlap
            answer_tokens = set(answer_lower.split())
            passage_tokens = set(passage.lower().split())
            overlap = len(answer_tokens & passage_tokens) / len(answer_tokens) if answer_tokens else 0
            labels.append(overlap)
    
    # Pad if necessary
    while len(labels) < num_passages:
        labels.append(0.0)
    
    return torch.tensor(labels[:num_passages], dtype=torch.float32)
