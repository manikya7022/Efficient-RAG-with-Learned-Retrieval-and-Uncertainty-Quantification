"""
Unit tests for the evaluation protocol module.

Run with: pytest tests/test_eval_protocol.py -v
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_uq.eval_protocol import (
    RAGEvaluator,
    RetrievalMetrics,
    GenerationMetrics,
    CalibrationMetrics,
    EfficiencyMetrics
)


class TestRetrievalMetrics:
    """Tests for retrieval evaluation."""
    
    @pytest.fixture
    def evaluator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield RAGEvaluator(output_dir=tmpdir)
    
    def test_recall_at_k_perfect(self, evaluator):
        """Test Recall@K for perfect retrieval."""
        retrieved = [["doc1", "doc2", "doc3"]]
        gold = [["doc1", "doc2"]]
        
        metrics = evaluator.evaluate_retrieval(retrieved, gold, k_values=[1, 2, 3])
        
        assert metrics.recall_at_k[1] == 0.5  # 1 out of 2
        assert metrics.recall_at_k[2] == 1.0  # 2 out of 2
        assert metrics.recall_at_k[3] == 1.0  # 2 out of 2
    
    def test_recall_at_k_zero(self, evaluator):
        """Test Recall@K when no relevant docs retrieved."""
        retrieved = [["doc4", "doc5", "doc6"]]
        gold = [["doc1", "doc2"]]
        
        metrics = evaluator.evaluate_retrieval(retrieved, gold, k_values=[3])
        
        assert metrics.recall_at_k[3] == 0.0
    
    def test_mrr_first_position(self, evaluator):
        """Test MRR when relevant doc is first."""
        retrieved = [["relevant", "other1", "other2"]]
        gold = [["relevant"]]
        
        metrics = evaluator.evaluate_retrieval(retrieved, gold)
        
        assert metrics.mrr == 1.0
    
    def test_mrr_third_position(self, evaluator):
        """Test MRR when relevant doc is third."""
        retrieved = [["other1", "other2", "relevant"]]
        gold = [["relevant"]]
        
        metrics = evaluator.evaluate_retrieval(retrieved, gold)
        
        assert metrics.mrr == 1/3
    
    def test_ndcg_perfect(self, evaluator):
        """Test NDCG@10 for perfect ranking."""
        retrieved = [["doc1", "doc2", "doc3"]]
        gold = [["doc1", "doc2", "doc3"]]
        
        metrics = evaluator.evaluate_retrieval(retrieved, gold)
        
        # Perfect ranking should have NDCG close to 1
        assert metrics.ndcg_at_10 > 0.9


class TestGenerationMetrics:
    """Tests for generation evaluation."""
    
    @pytest.fixture
    def evaluator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield RAGEvaluator(output_dir=tmpdir)
    
    def test_exact_match_identical(self, evaluator):
        """Test EM for identical answers."""
        predictions = ["The answer is 42"]
        references = ["The answer is 42"]
        
        metrics = evaluator.evaluate_generation(predictions, references)
        
        assert metrics.exact_match == 1.0
    
    def test_exact_match_normalized(self, evaluator):
        """Test EM with normalization."""
        predictions = ["The Answer Is 42!"]
        references = ["the answer is 42"]
        
        metrics = evaluator.evaluate_generation(predictions, references)
        
        assert metrics.exact_match == 1.0
    
    def test_exact_match_different(self, evaluator):
        """Test EM for different answers."""
        predictions = ["The answer is 41"]
        references = ["The answer is 42"]
        
        metrics = evaluator.evaluate_generation(predictions, references)
        
        assert metrics.exact_match < 1.0
    
    def test_token_f1_identical(self, evaluator):
        """Test F1 for identical answers."""
        predictions = ["the answer is 42"]
        references = ["the answer is 42"]
        
        metrics = evaluator.evaluate_generation(predictions, references)
        
        assert metrics.f1 == 1.0
    
    def test_token_f1_partial(self, evaluator):
        """Test F1 for partial overlap."""
        predictions = ["the answer"]
        references = ["the answer is 42"]
        
        metrics = evaluator.evaluate_generation(predictions, references)
        
        assert 0 < metrics.f1 < 1
    
    def test_rouge_l(self, evaluator):
        """Test ROUGE-L computation."""
        predictions = ["The quick brown fox"]
        references = ["The quick brown dog"]
        
        metrics = evaluator.evaluate_generation(predictions, references)
        
        # Should have partial ROUGE score
        assert 0 < metrics.rouge_l < 1


class TestCalibrationMetrics:
    """Tests for calibration evaluation."""
    
    @pytest.fixture
    def evaluator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield RAGEvaluator(output_dir=tmpdir, n_bins=5)
    
    def test_ece_perfectly_calibrated(self, evaluator):
        """Test ECE for perfectly calibrated predictions."""
        # Confidence matches accuracy in each bin
        confidences = [0.1, 0.3, 0.5, 0.7, 0.9] * 20
        correctness = [0, 0, 0, 1, 1] * 20  # Roughly matching
        
        metrics = evaluator.evaluate_calibration(confidences, correctness)
        
        # Should have low ECE
        assert metrics.ece < 0.5
    
    def test_ece_overconfident(self, evaluator):
        """Test ECE for overconfident model."""
        # High confidence, low accuracy
        confidences = [0.9] * 100
        correctness = [0] * 80 + [1] * 20  # Only 20% correct
        
        metrics = evaluator.evaluate_calibration(confidences, correctness)
        
        # Should have high ECE (overconfidence)
        assert metrics.ece > 0.5 or metrics.mce > 0.5
    
    def test_brier_score_perfect(self, evaluator):
        """Test Brier score for perfect predictions."""
        # Perfect: conf=1 when correct, conf=0 when wrong
        confidences = [1.0, 0.0, 1.0, 0.0]
        correctness = [1, 0, 1, 0]
        
        metrics = evaluator.evaluate_calibration(confidences, correctness)
        
        assert metrics.brier_score == 0.0
    
    def test_brier_score_worst(self, evaluator):
        """Test Brier score for worst predictions."""
        # Worst: conf=1 when wrong, conf=0 when correct
        confidences = [0.0, 1.0, 0.0, 1.0]
        correctness = [1, 0, 1, 0]
        
        metrics = evaluator.evaluate_calibration(confidences, correctness)
        
        assert metrics.brier_score == 1.0


class TestEfficiencyMetrics:
    """Tests for efficiency evaluation."""
    
    @pytest.fixture
    def evaluator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield RAGEvaluator(output_dir=tmpdir)
    
    def test_latency_computation(self, evaluator):
        """Test latency metric computation."""
        latencies = [0.1, 0.2, 0.3, 0.4, 0.5]  # seconds
        
        metrics = evaluator.evaluate_efficiency(latencies)
        
        assert metrics.avg_latency_ms == 300.0  # 0.3s = 300ms
        assert metrics.p50_latency_ms == 300.0
        assert metrics.p95_latency_ms >= 400.0
    
    def test_throughput(self, evaluator):
        """Test throughput computation."""
        latencies = [0.1] * 10  # 100ms each
        
        metrics = evaluator.evaluate_efficiency(latencies)
        
        # Should be ~10 QPS
        assert metrics.throughput_qps == pytest.approx(10.0, rel=0.1)


class TestVisualization:
    """Tests for plotting functions."""
    
    @pytest.fixture
    def evaluator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield RAGEvaluator(output_dir=tmpdir)
    
    def test_reliability_diagram(self, evaluator):
        """Test reliability diagram generation."""
        metrics = CalibrationMetrics(
            ece=0.1,
            mce=0.2,
            brier_score=0.15,
            confidence_per_bin=[0.1, 0.3, 0.5, 0.7, 0.9],
            accuracy_per_bin=[0.15, 0.28, 0.52, 0.68, 0.88],
            bin_counts=[20, 25, 30, 15, 10]
        )
        
        path = evaluator.plot_reliability_diagram(metrics)
        
        if path:  # matplotlib may not be available
            assert Path(path).exists()


class TestBootstrap:
    """Tests for bootstrap confidence intervals."""
    
    @pytest.fixture
    def evaluator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield RAGEvaluator(output_dir=tmpdir)
    
    def test_bootstrap_metric(self, evaluator):
        """Test bootstrap CI computation."""
        data = list(range(100))
        
        point, lower, upper = evaluator.bootstrap_metric(
            data,
            metric_fn=lambda x: np.mean(x),
            n_bootstrap=100
        )
        
        assert lower <= point <= upper
        assert point == pytest.approx(49.5, rel=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
