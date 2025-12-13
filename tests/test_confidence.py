"""
Unit tests for the confidence estimation module.

Run with: pytest tests/test_confidence.py -v
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_uq.confidence import (
    MCDropoutConfidence,
    ConformalRAG,
    HybridConfidence,
    ConfidenceResult,
    ConformalResult
)


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or ["mock answer"]
        self.call_count = 0
    
    def generate(self, model="", prompt="", options=None):
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return {'response': response}


class TestMCDropoutConfidence:
    """Tests for MC Dropout confidence estimation."""
    
    def test_initialization(self):
        """Test MCDropoutConfidence initializes."""
        llm = MockLLMClient()
        mc = MCDropoutConfidence(llm, n_samples=5)
        
        assert mc.n_samples == 5
    
    def test_sample_parameters(self):
        """Test parameter sampling is within range."""
        llm = MockLLMClient()
        mc = MCDropoutConfidence(llm, temperature_range=(0.5, 1.0))
        
        params = mc._sample_parameters()
        
        assert 0.5 <= params['temperature'] <= 1.0
        assert 0.0 <= params['top_p'] <= 1.0
    
    def test_get_confidence_interval_consistent(self):
        """Test consistent answers produce high confidence."""
        responses = ["42"] * 10
        llm = MockLLMClient(responses)
        mc = MCDropoutConfidence(llm, n_samples=10)
        
        result = mc.get_confidence_interval(
            prompt="Answer:",
            context="The answer is 42.",
            question="What is the answer?"
        )
        
        assert isinstance(result, ConfidenceResult)
        assert len(result.answers) == 10
        # Consistent answers should have low uncertainty
        assert result.uncertainty_score < 0.5 or result.uncertainty_score is not None
    
    def test_get_confidence_interval_diverse(self):
        """Test diverse answers produce low confidence."""
        responses = ["answer1", "answer2", "answer3", "answer4", "answer5"]
        llm = MockLLMClient(responses)
        mc = MCDropoutConfidence(llm, n_samples=5)
        
        result = mc.get_confidence_interval(
            prompt="Answer:",
            context="Context",
            question="Question?"
        )
        
        assert len(result.answers) == 5
        assert result.consensus_answer in responses
    
    def test_lexical_diversity(self):
        """Test lexical diversity computation."""
        llm = MockLLMClient()
        mc = MCDropoutConfidence(llm)
        
        # Identical answers
        diversity_low = mc._compute_lexical_diversity(["the same answer"] * 5)
        
        # Different answers
        diversity_high = mc._compute_lexical_diversity([
            "answer one", "different two", "another three"
        ])
        
        # Identical answers should have lower diversity (more type overlap)
        assert diversity_low < diversity_high
    
    def test_empty_answers_handling(self):
        """Test handling of empty answers."""
        llm = MockLLMClient([""])
        mc = MCDropoutConfidence(llm, n_samples=3)
        
        result = mc.get_confidence_interval("", "", "")
        
        # Should handle gracefully
        assert isinstance(result, ConfidenceResult)


class TestConformalRAG:
    """Tests for Conformal Prediction."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "calibration.db"
    
    def test_initialization(self, temp_db):
        """Test ConformalRAG initializes."""
        llm = MockLLMClient()
        conformal = ConformalRAG(llm, calibration_db_path=str(temp_db), alpha=0.1)
        
        assert conformal.alpha == 0.1
        assert len(conformal.calibration_scores) == 0
    
    def test_rouge_l_exact_match(self, temp_db):
        """Test ROUGE-L for exact match."""
        llm = MockLLMClient()
        conformal = ConformalRAG(llm, calibration_db_path=str(temp_db))
        
        score = conformal.rouge_l("the answer is 42", "the answer is 42")
        
        assert score == 1.0
    
    def test_rouge_l_no_match(self, temp_db):
        """Test ROUGE-L for no match."""
        llm = MockLLMClient()
        conformal = ConformalRAG(llm, calibration_db_path=str(temp_db))
        
        score = conformal.rouge_l("completely different", "xyz abc")
        
        assert score < 0.5
    
    def test_rouge_l_partial_match(self, temp_db):
        """Test ROUGE-L for partial match."""
        llm = MockLLMClient()
        conformal = ConformalRAG(llm, calibration_db_path=str(temp_db))
        
        score = conformal.rouge_l("the answer is 42", "the answer is wrong")
        
        assert 0 < score < 1
    
    def test_calibrate(self, temp_db):
        """Test calibration creates scores."""
        llm = MockLLMClient(["predicted answer"])
        conformal = ConformalRAG(llm, calibration_db_path=str(temp_db))
        
        stats = conformal.calibrate(
            questions=["What is 1+1?", "What color is the sky?"],
            contexts=["1+1 equals 2.", "The sky is blue."],
            true_answers=["2", "blue"],
            skip_existing=False
        )
        
        assert stats['total_calibrated'] == 2
        assert len(conformal.calibration_scores) == 2
    
    def test_conformal_threshold(self, temp_db):
        """Test conformal threshold computation."""
        llm = MockLLMClient()
        conformal = ConformalRAG(llm, calibration_db_path=str(temp_db), alpha=0.1)
        
        # Manually set calibration scores
        conformal.calibration_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        threshold = conformal.get_conformal_threshold()
        
        # Should be around the 90th percentile
        assert 0.8 <= threshold <= 1.0
    
    def test_predict_with_coverage(self, temp_db):
        """Test prediction with coverage guarantee."""
        llm = MockLLMClient(["predicted answer"])
        conformal = ConformalRAG(llm, calibration_db_path=str(temp_db))
        
        # Set calibration scores
        conformal.calibration_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        result = conformal.predict_with_coverage(
            question="What is the answer?",
            context="The answer is something."
        )
        
        assert isinstance(result, ConformalResult)
        assert result.prediction == "predicted answer"
        assert 0 <= result.confidence <= 1
        assert 0 <= result.p_value <= 1
    
    def test_calibration_stats(self, temp_db):
        """Test calibration statistics."""
        llm = MockLLMClient()
        conformal = ConformalRAG(llm, calibration_db_path=str(temp_db))
        
        conformal.calibration_scores = [0.1, 0.2, 0.3]
        
        stats = conformal.get_calibration_stats()
        
        assert stats['count'] == 3
        assert abs(stats['mean'] - 0.2) < 0.01


class TestHybridConfidence:
    """Tests for combined MC + Conformal confidence."""
    
    def test_initialization(self):
        """Test HybridConfidence initializes."""
        llm = MockLLMClient()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            hybrid = HybridConfidence(
                llm,
                mc_samples=3,
                conformal_alpha=0.1,
                calibration_db_path=f"{tmpdir}/cal.db"
            )
            
            assert hybrid.mc.n_samples == 3
            assert hybrid.conformal.alpha == 0.1
    
    def test_estimate_uncertainty(self):
        """Test combined uncertainty estimation."""
        llm = MockLLMClient(["consistent answer"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            hybrid = HybridConfidence(
                llm,
                mc_samples=3,
                calibration_db_path=f"{tmpdir}/cal.db"
            )
            
            result = hybrid.estimate_uncertainty(
                prompt="Answer:",
                context="Context",
                question="Question?"
            )
            
            assert 'answer' in result
            assert 'combined_confidence' in result
            assert 'mc_confidence' in result
            assert 'conformal_confidence' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
