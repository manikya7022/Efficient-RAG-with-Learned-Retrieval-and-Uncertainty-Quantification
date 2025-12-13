"""
Unit tests for the RetrievalRouter module.

Run with: pytest tests/test_router.py -v
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_uq.router import (
    RetrievalRouter, 
    RouterConfig, 
    ApproxNDCGLoss,
    RouterTrainer,
    create_pseudo_labels
)


class TestRetrievalRouter:
    """Tests for RetrievalRouter class."""
    
    def test_initialization(self):
        """Test router initializes with correct config."""
        config = RouterConfig(hidden_dim=32, dropout=0.1)
        router = RetrievalRouter(config)
        
        assert router.config.hidden_dim == 32
        assert router.config.dropout == 0.1
        assert router._count_params() > 0
    
    def test_default_config(self):
        """Test default configuration."""
        router = RetrievalRouter()
        
        assert router.config.hidden_dim == 64
        assert router.config.dropout == 0.1
    
    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        router = RetrievalRouter()
        
        batch_size = 4
        num_passages = 20
        
        bm25_scores = torch.randn(batch_size, num_passages)
        dense_scores = torch.randn(batch_size, num_passages)
        
        weights = router(bm25_scores, dense_scores)
        
        assert weights.shape == (batch_size, num_passages)
        assert (weights >= 0).all()
        assert (weights <= 1).all()
    
    def test_forward_single_batch(self):
        """Test forward with batch size 1."""
        router = RetrievalRouter()
        
        bm25_scores = torch.randn(1, 10)
        dense_scores = torch.randn(1, 10)
        
        weights = router(bm25_scores, dense_scores)
        
        assert weights.shape == (1, 10)
    
    def test_hybrid_rerank(self):
        """Test hybrid reranking returns correct top-k."""
        router = RetrievalRouter()
        
        bm25_scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        dense_scores = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
        
        scores, indices = router.hybrid_rerank(bm25_scores, dense_scores, top_k=3)
        
        assert scores.shape == (1, 3)
        assert indices.shape == (1, 3)
    
    def test_hybrid_rerank_top_k_larger_than_passages(self):
        """Test hybrid rerank when top_k > num_passages."""
        router = RetrievalRouter()
        
        bm25_scores = torch.randn(1, 5)
        dense_scores = torch.randn(1, 5)
        
        scores, indices = router.hybrid_rerank(bm25_scores, dense_scores, top_k=10)
        
        assert scores.shape == (1, 5)
    
    def test_get_routing_decision(self):
        """Test routing decision analysis."""
        router = RetrievalRouter()
        
        bm25_scores = torch.randn(2, 10)
        dense_scores = torch.randn(2, 10)
        
        decision = router.get_routing_decision(bm25_scores, dense_scores)
        
        assert 'avg_dense_weight' in decision
        assert 'weight_std' in decision
        assert 'dense_preferred_ratio' in decision
        assert 'bm25_preferred_ratio' in decision
        assert 'routing_weights' in decision
    
    def test_training_mode(self):
        """Test router updates statistics during training."""
        router = RetrievalRouter()
        router.train()
        
        bm25_scores = torch.randn(8, 20)
        dense_scores = torch.randn(8, 20)
        
        # Run forward to update stats
        _ = router(bm25_scores, dense_scores, update_stats=True)
        
        assert router.stats_initialized
    
    def test_eval_mode(self):
        """Test router doesn't update stats during eval."""
        router = RetrievalRouter()
        router.eval()
        
        bm25_scores = torch.randn(8, 20)
        dense_scores = torch.randn(8, 20)
        
        # Run forward without updating
        _ = router(bm25_scores, dense_scores, update_stats=False)


class TestApproxNDCGLoss:
    """Tests for ApproxNDCG loss function."""
    
    def test_initialization(self):
        """Test loss function initializes."""
        loss_fn = ApproxNDCGLoss(temperature=1.0)
        assert loss_fn.temperature == 1.0
    
    def test_perfect_ranking(self):
        """Test loss is low for perfect ranking."""
        loss_fn = ApproxNDCGLoss()
        
        # Perfect ranking: scores match relevance
        predicted = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        relevance = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
        
        loss = loss_fn(predicted, relevance)
        
        # Loss should be close to -1 (negative NDCG)
        assert loss < 0
    
    def test_inverted_ranking(self):
        """Test loss is high for inverted ranking."""
        loss_fn = ApproxNDCGLoss()
        
        # Inverted ranking: low scores for relevant items
        predicted = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        relevance = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
        
        loss_inverted = loss_fn(predicted, relevance)
        
        # Perfect ranking for comparison
        predicted_good = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        loss_good = loss_fn(predicted_good, relevance)
        
        # Inverted should have higher loss (less negative)
        assert loss_inverted > loss_good
    
    def test_batch_processing(self):
        """Test loss works with batched inputs."""
        loss_fn = ApproxNDCGLoss()
        
        batch_size = 8
        num_items = 10
        
        predicted = torch.randn(batch_size, num_items)
        relevance = torch.rand(batch_size, num_items)
        
        loss = loss_fn(predicted, relevance)
        
        assert loss.dim() == 0  # Scalar
        assert torch.isfinite(loss)
    
    def test_with_mask(self):
        """Test loss with padding mask."""
        loss_fn = ApproxNDCGLoss()
        
        predicted = torch.randn(2, 5)
        relevance = torch.rand(2, 5)
        mask = torch.tensor([[True, True, True, False, False],
                            [True, True, True, True, False]])
        
        loss = loss_fn(predicted, relevance, mask=mask)
        
        assert torch.isfinite(loss)
    
    def test_temperature_effect(self):
        """Test that temperature affects loss smoothness."""
        loss_high_temp = ApproxNDCGLoss(temperature=2.0)
        loss_low_temp = ApproxNDCGLoss(temperature=0.5)
        
        predicted = torch.randn(4, 10)
        relevance = torch.rand(4, 10)
        
        # Both should produce valid losses
        l1 = loss_high_temp(predicted, relevance)
        l2 = loss_low_temp(predicted, relevance)
        
        assert torch.isfinite(l1)
        assert torch.isfinite(l2)


class TestRouterTrainer:
    """Tests for RouterTrainer class."""
    
    def test_initialization(self):
        """Test trainer initializes correctly."""
        router = RetrievalRouter()
        trainer = RouterTrainer(router, learning_rate=1e-3)
        
        assert trainer.router is router
        assert len(trainer.train_losses) == 0
    
    def test_train_epoch(self):
        """Test single training epoch."""
        router = RetrievalRouter()
        trainer = RouterTrainer(router)
        
        # Create synthetic data
        bm25 = torch.randn(16, 20)
        dense = torch.randn(16, 20)
        relevance = torch.rand(16, 20)
        
        loss = trainer.train_epoch((bm25, dense, relevance))
        
        assert isinstance(loss, float)
        assert loss < 0  # Negative NDCG
    
    def test_validate(self):
        """Test validation pass."""
        router = RetrievalRouter()
        trainer = RouterTrainer(router)
        
        bm25 = torch.randn(8, 20)
        dense = torch.randn(8, 20)
        relevance = torch.rand(8, 20)
        
        val_loss = trainer.validate((bm25, dense, relevance))
        
        assert isinstance(val_loss, float)
    
    def test_fit_convergence(self):
        """Test that training reduces loss."""
        router = RetrievalRouter(RouterConfig(hidden_dim=16))
        trainer = RouterTrainer(router, learning_rate=1e-2)
        
        # Create synthetic data where router can learn
        np.random.seed(42)
        n_samples = 100
        n_passages = 10
        
        bm25 = torch.randn(n_samples, n_passages)
        dense = torch.randn(n_samples, n_passages)
        
        # Relevance that correlates with dense scores
        relevance = (dense > 0).float() * 0.8 + torch.rand(n_samples, n_passages) * 0.2
        
        history = trainer.fit(
            train_data=(bm25, dense, relevance),
            num_epochs=5,
            batch_size=32
        )
        
        # Check loss decreased
        assert len(history['train_losses']) == 5
        assert history['train_losses'][-1] <= history['train_losses'][0]


class TestPseudoLabels:
    """Tests for pseudo-label creation."""
    
    def test_exact_match(self):
        """Test pseudo-labels for exact match."""
        passages = ["The answer is 42.", "Something else."]
        answer = "42"
        
        labels = create_pseudo_labels(passages, passages, answer, num_passages=2)
        
        assert labels[0] > 0  # First passage contains answer
    
    def test_no_match(self):
        """Test pseudo-labels when no match."""
        passages = ["Some text.", "Other text."]
        answer = "completely different"
        
        labels = create_pseudo_labels(passages, passages, answer, num_passages=2)
        
        assert labels[0] < 1.0  # No exact match
    
    def test_padding(self):
        """Test pseudo-labels are padded correctly."""
        passages = ["Text."]
        answer = "answer"
        
        labels = create_pseudo_labels(passages, passages, answer, num_passages=5)
        
        assert len(labels) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
