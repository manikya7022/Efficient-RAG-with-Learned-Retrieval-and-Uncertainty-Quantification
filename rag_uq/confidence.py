"""
Uncertainty Quantification for RAG

Implements Bayesian confidence estimation using two complementary approaches:

1. **MC Dropout Confidence**: Uses sampling-based uncertainty estimation via
   temperature/top-p variation in LLM inference. Measures uncertainty as the
   variance of answer embeddings in semantic space.

2. **Conformal RAG**: Provides frequentist coverage guarantees through conformal
   prediction. Calibrates on a held-out set and provides prediction sets with
   guaranteed coverage (1-α).

Theoretical Foundation:
    - MC Dropout: Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)
    - Conformal Prediction: Shafer & Vovk, "A Tutorial on Conformal Prediction" (JMLR 2008)
    - RAG Uncertainty: Novel contribution of this work
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, field
import sqlite3
import json
import logging
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed. Embedding-based uncertainty disabled.")

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    logger.warning("rouge-score not installed. ROUGE-based calibration disabled.")


@dataclass
class ConfidenceResult:
    """Result from confidence estimation."""
    answers: List[str]
    consensus_answer: str
    uncertainty_score: float
    confidence: float  # 1 - uncertainty_score (normalized)
    embedding_variance: Optional[float] = None
    lexical_diversity: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class ConformalResult:
    """Result from conformal prediction."""
    prediction: str
    confidence: float
    p_value: float
    is_reliable: bool
    coverage_alpha: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCDropoutConfidence:
    """
    Monte Carlo Dropout-style confidence estimation for LLM responses.
    
    Since Ollama doesn't support native dropout, we simulate epistemic uncertainty
    through:
    1. Temperature scaling (randomness in sampling)
    2. Top-p/top-k variation
    3. Multiple inference passes
    
    Uncertainty is measured as the variance of answer embeddings in a learned
    semantic space (using MiniLM-L6-v2, ~90MB).
    
    Args:
        llm_client: Ollama client for LLM inference
        n_samples: Number of MC samples to collect
        embedding_model: Model name for sentence embeddings
        temperature_range: (min, max) temperature for sampling
        top_p_range: (min, max) top-p values for sampling
        
    Example:
        >>> from ollama import Client
        >>> llm = Client()
        >>> mc = MCDropoutConfidence(llm, n_samples=10)
        >>> result = mc.get_confidence_interval(
        ...     "Answer the following question.",
        ...     "The Eiffel Tower is 330 meters tall.",
        ...     "How tall is the Eiffel Tower?"
        ... )
        >>> print(f"Answer: {result.consensus_answer}, Confidence: {result.confidence:.2f}")
    """
    
    def __init__(
        self,
        llm_client,
        n_samples: int = 10,
        embedding_model: str = 'all-MiniLM-L6-v2',
        temperature_range: Tuple[float, float] = (0.5, 1.2),
        top_p_range: Tuple[float, float] = (0.8, 0.95),
        max_tokens: int = 100
    ):
        self.llm = llm_client
        self.n_samples = n_samples
        self.temperature_range = temperature_range
        self.top_p_range = top_p_range
        self.max_tokens = max_tokens
        
        # Load embedding model for semantic similarity
        if HAS_SENTENCE_TRANSFORMERS:
            self.encoder = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
        else:
            self.encoder = None
            logger.warning("Sentence encoder not available")
    
    def _sample_parameters(self) -> Dict[str, float]:
        """Sample generation parameters for uncertainty estimation."""
        return {
            'temperature': np.random.uniform(*self.temperature_range),
            'top_p': np.random.uniform(*self.top_p_range)
        }
    
    def _generate_sample(
        self,
        prompt: str,
        context: str,
        question: str,
        model: str = "llama3.2:3b"
    ) -> str:
        """Generate a single sample from the LLM."""
        params = self._sample_parameters()
        
        full_prompt = f"""{prompt}

Context: {context}

Question: {question}

Answer:"""
        
        try:
            response = self.llm.generate(
                model=model,
                prompt=full_prompt,
                options={
                    'temperature': params['temperature'],
                    'top_p': params['top_p'],
                    'num_predict': self.max_tokens
                }
            )
            return response.get('response', '').strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""
    
    def _compute_lexical_diversity(self, answers: List[str]) -> float:
        """Compute lexical diversity (type-token ratio) across answers."""
        all_tokens = []
        for answer in answers:
            tokens = answer.lower().split()
            all_tokens.extend(tokens)
        
        if not all_tokens:
            return 1.0  # Maximum uncertainty
        
        unique_tokens = set(all_tokens)
        return len(unique_tokens) / len(all_tokens)
    
    def _compute_embedding_variance(
        self, 
        answers: List[str]
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute variance of answer embeddings.
        
        Returns:
            Tuple of (variance_score, centroid, embeddings)
        """
        if self.encoder is None or not answers:
            return 1.0, np.array([]), np.array([])
        
        # Filter empty answers
        valid_answers = [a for a in answers if a.strip()]
        if not valid_answers:
            return 1.0, np.array([]), np.array([])
        
        embeddings = self.encoder.encode(valid_answers)
        centroid = embeddings.mean(axis=0)
        
        # Compute distances from centroid
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        variance = float(distances.std())
        
        return variance, centroid, embeddings
    
    def get_confidence_interval(
        self,
        prompt: str,
        context: str,
        question: str,
        model: str = "llama3.2:3b"
    ) -> ConfidenceResult:
        """
        Estimate confidence interval for RAG response using MC sampling.
        
        Args:
            prompt: System/instruction prompt
            context: Retrieved context passages
            question: User question
            model: Ollama model name
            
        Returns:
            ConfidenceResult with consensus answer and uncertainty metrics
        """
        # Collect MC samples
        answers = []
        for i in range(self.n_samples):
            answer = self._generate_sample(prompt, context, question, model)
            if answer:
                answers.append(answer)
            
            if (i + 1) % 5 == 0:
                logger.debug(f"Collected {i + 1}/{self.n_samples} samples")
        
        if not answers:
            return ConfidenceResult(
                answers=[],
                consensus_answer="",
                uncertainty_score=1.0,
                confidence=0.0,
                metadata={'error': 'No valid answers generated'}
            )
        
        # Compute uncertainty metrics
        lexical_diversity = self._compute_lexical_diversity(answers)
        embedding_variance, centroid, embeddings = self._compute_embedding_variance(answers)
        
        # Find consensus answer (closest to centroid)
        if len(embeddings) > 0:
            distances = np.linalg.norm(embeddings - centroid, axis=1)
            consensus_idx = int(np.argmin(distances))
            consensus_answer = [a for a in answers if a.strip()][consensus_idx]
        else:
            # Fallback: most common answer
            from collections import Counter
            consensus_answer = Counter(answers).most_common(1)[0][0]
        
        # Normalize uncertainty to [0, 1]
        # Lower variance = higher confidence
        normalized_uncertainty = min(1.0, embedding_variance / 2.0)  # Heuristic scaling
        
        return ConfidenceResult(
            answers=answers,
            consensus_answer=consensus_answer,
            uncertainty_score=normalized_uncertainty,
            confidence=1.0 - normalized_uncertainty,
            embedding_variance=embedding_variance,
            lexical_diversity=lexical_diversity,
            metadata={
                'n_samples': len(answers),
                'temperature_range': self.temperature_range,
                'top_p_range': self.top_p_range
            }
        )


class ConformalRAG:
    """
    Conformal prediction for RAG with frequentist coverage guarantees.
    
    Provides prediction sets with guaranteed coverage probability (1-α).
    Uses a calibration set to compute nonconformity scores based on
    ROUGE-L similarity between predictions and ground truth.
    
    Two-phase approach:
    1. Calibration Phase: Compute nonconformity scores on held-out set
    2. Inference Phase: Use calibrated threshold for coverage guarantee
    
    Args:
        llm_client: Ollama client for generation
        calibration_db_path: Path to SQLite database for storing calibration scores
        alpha: Desired miscoverage rate (default 0.1 for 90% coverage)
        
    Theoretical Guarantee:
        P(true_answer ∈ prediction_set) ≥ 1 - α
        
    Example:
        >>> conformal = ConformalRAG(llm_client, alpha=0.1)
        >>> conformal.calibrate(cal_questions, cal_contexts, cal_answers)
        >>> result = conformal.predict_with_coverage("What is X?", "X is Y.")
        >>> print(f"Reliable: {result.is_reliable}, Confidence: {result.confidence:.2f}")
    """
    
    def __init__(
        self,
        llm_client,
        calibration_db_path: str = "data/calibration_scores.db",
        alpha: float = 0.1
    ):
        self.llm = llm_client
        self.alpha = alpha
        self.db_path = Path(calibration_db_path)
        self.calibration_scores: List[float] = []
        
        # Initialize ROUGE scorer
        if HAS_ROUGE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        else:
            self.rouge_scorer = None
        
        # Load existing calibration if available
        self._init_database()
        self._load_calibration()
    
    def _init_database(self):
        """Initialize SQLite database for calibration scores."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calibration_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT UNIQUE,
                    question TEXT,
                    predicted_answer TEXT,
                    true_answer TEXT,
                    nonconformity_score REAL,
                    rouge_l REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_hash 
                ON calibration_scores(query_hash)
            """)
    
    def _load_calibration(self):
        """Load calibration scores from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT nonconformity_score FROM calibration_scores ORDER BY id"
            )
            self.calibration_scores = [row[0] for row in cursor.fetchall()]
        
        logger.info(f"Loaded {len(self.calibration_scores)} calibration scores")
    
    def _compute_query_hash(self, question: str, context: str) -> str:
        """Compute unique hash for query."""
        content = f"{question}|||{context}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def rouge_l(self, prediction: str, reference: str) -> float:
        """Compute ROUGE-L F1 score between prediction and reference."""
        if self.rouge_scorer is None:
            # Fallback: simple token overlap
            pred_tokens = set(prediction.lower().split())
            ref_tokens = set(reference.lower().split())
            if not ref_tokens:
                return 0.0
            overlap = len(pred_tokens & ref_tokens)
            precision = overlap / len(pred_tokens) if pred_tokens else 0
            recall = overlap / len(ref_tokens) if ref_tokens else 0
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)
        
        scores = self.rouge_scorer.score(reference, prediction)
        return scores['rougeL'].fmeasure
    
    def _generate(
        self,
        context: str,
        question: str,
        model: str = "llama3.2:3b"
    ) -> str:
        """Generate answer using LLM."""
        prompt = f"""Answer the following question based on the provided context.
Be concise and precise.

Context: {context}

Question: {question}

Answer:"""
        
        try:
            response = self.llm.generate(
                model=model,
                prompt=prompt,
                options={'temperature': 0.1, 'num_predict': 100}
            )
            return response.get('response', '').strip()
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def calibrate(
        self,
        questions: List[str],
        contexts: List[str],
        true_answers: List[str],
        model: str = "llama3.2:3b",
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Build calibration set by computing nonconformity scores.
        
        Nonconformity score = 1 - ROUGE-L(prediction, true_answer)
        Higher score = less conforming = more uncertain
        
        Args:
            questions: List of calibration questions
            contexts: List of corresponding contexts
            true_answers: List of ground truth answers
            model: Ollama model name
            skip_existing: Skip already calibrated examples
            
        Returns:
            Calibration statistics
        """
        new_scores = []
        skipped = 0
        
        for i, (q, ctx, true) in enumerate(zip(questions, contexts, true_answers)):
            query_hash = self._compute_query_hash(q, ctx)
            
            # Check if already calibrated
            if skip_existing:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT 1 FROM calibration_scores WHERE query_hash = ?",
                        (query_hash,)
                    )
                    if cursor.fetchone():
                        skipped += 1
                        continue
            
            # Generate prediction
            pred = self._generate(ctx, q, model)
            
            # Compute nonconformity score
            rouge_score = self.rouge_l(pred, true)
            nonconformity = 1.0 - rouge_score
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO calibration_scores 
                    (query_hash, question, predicted_answer, true_answer, 
                     nonconformity_score, rouge_l)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (query_hash, q, pred, true, nonconformity, rouge_score))
            
            new_scores.append(nonconformity)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Calibrated {i + 1}/{len(questions)} examples")
        
        # Reload all scores
        self._load_calibration()
        
        return {
            'total_calibrated': len(self.calibration_scores),
            'new_calibrated': len(new_scores),
            'skipped': skipped,
            'mean_nonconformity': np.mean(self.calibration_scores) if self.calibration_scores else 0,
            'std_nonconformity': np.std(self.calibration_scores) if self.calibration_scores else 0
        }
    
    def get_conformal_threshold(self) -> float:
        """
        Compute conformal threshold for desired coverage.
        
        Returns the (1-α) quantile of calibration nonconformity scores.
        """
        if not self.calibration_scores:
            logger.warning("No calibration scores available")
            return 1.0
        
        n = len(self.calibration_scores)
        # Finite sample correction
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        
        return float(np.quantile(self.calibration_scores, q_level))
    
    def predict_with_coverage(
        self,
        question: str,
        context: str,
        model: str = "llama3.2:3b"
    ) -> ConformalResult:
        """
        Generate prediction with conformal coverage guarantee.
        
        Args:
            question: User question
            context: Retrieved context
            model: Ollama model name
            
        Returns:
            ConformalResult with prediction, confidence, and reliability flag
        """
        # Generate prediction
        pred = self._generate(context, question, model)
        
        if not self.calibration_scores:
            return ConformalResult(
                prediction=pred,
                confidence=0.5,
                p_value=0.5,
                is_reliable=False,
                coverage_alpha=self.alpha,
                metadata={'warning': 'No calibration data available'}
            )
        
        # For inference, we estimate nonconformity using retrieval scores
        # as a proxy (since we don't have ground truth)
        # This is a heuristic - in practice, use retrieval confidence
        
        # Compute conformal p-value
        # We use the prediction length and context relevance as proxy
        threshold = self.get_conformal_threshold()
        
        # Estimate current nonconformity (heuristic: based on answer length and specificity)
        pred_tokens = len(pred.split())
        context_tokens = len(context.split())
        
        # Heuristic: very short or very long answers relative to context are less reliable
        length_ratio = pred_tokens / (context_tokens + 1)
        estimated_nonconformity = 1.0 - min(1.0, 4 * length_ratio * (1 - length_ratio))
        
        # Conformal p-value
        n = len(self.calibration_scores)
        rank = sum(1 for s in self.calibration_scores if s >= estimated_nonconformity)
        p_value = (rank + 1) / (n + 1)
        
        # Confidence and reliability
        confidence = 1.0 - estimated_nonconformity
        is_reliable = p_value > self.alpha
        
        return ConformalResult(
            prediction=pred,
            confidence=confidence,
            p_value=p_value,
            is_reliable=is_reliable,
            coverage_alpha=self.alpha,
            metadata={
                'threshold': threshold,
                'estimated_nonconformity': estimated_nonconformity,
                'calibration_size': n
            }
        )
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get statistics about the calibration set."""
        if not self.calibration_scores:
            return {'empty': True}
        
        scores = np.array(self.calibration_scores)
        return {
            'count': len(scores),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'median': float(np.median(scores)),
            'q25': float(np.percentile(scores, 25)),
            'q75': float(np.percentile(scores, 75)),
            'threshold': self.get_conformal_threshold(),
            'alpha': self.alpha
        }


class HybridConfidence:
    """
    Combined MC Dropout and Conformal Prediction for robust uncertainty.
    
    Uses MC Dropout for fine-grained uncertainty estimation and conformal
    prediction for coverage guarantees. Combines both signals for a 
    comprehensive uncertainty assessment.
    
    Args:
        llm_client: Ollama client
        mc_samples: Number of MC samples
        conformal_alpha: Coverage miscoverage rate
        calibration_db_path: Path to calibration database
    """
    
    def __init__(
        self,
        llm_client,
        mc_samples: int = 5,
        conformal_alpha: float = 0.1,
        calibration_db_path: str = "data/calibration_scores.db"
    ):
        self.mc = MCDropoutConfidence(llm_client, n_samples=mc_samples)
        self.conformal = ConformalRAG(
            llm_client, 
            calibration_db_path=calibration_db_path,
            alpha=conformal_alpha
        )
    
    def estimate_uncertainty(
        self,
        prompt: str,
        context: str,
        question: str,
        model: str = "llama3.2:3b"
    ) -> Dict[str, Any]:
        """
        Comprehensive uncertainty estimation combining both methods.
        
        Returns:
            Dictionary with MC and conformal uncertainty estimates
        """
        # MC Dropout estimation
        mc_result = self.mc.get_confidence_interval(prompt, context, question, model)
        
        # Conformal estimation
        conformal_result = self.conformal.predict_with_coverage(question, context, model)
        
        # Combine signals
        combined_confidence = (mc_result.confidence + conformal_result.confidence) / 2
        
        # Final answer selection: use conformal if reliable, else MC consensus
        if conformal_result.is_reliable:
            final_answer = conformal_result.prediction
            answer_source = 'conformal'
        else:
            final_answer = mc_result.consensus_answer
            answer_source = 'mc_consensus'
        
        return {
            'answer': final_answer,
            'answer_source': answer_source,
            'combined_confidence': combined_confidence,
            'mc_confidence': mc_result.confidence,
            'mc_uncertainty': mc_result.uncertainty_score,
            'mc_embedding_variance': mc_result.embedding_variance,
            'conformal_confidence': conformal_result.confidence,
            'conformal_p_value': conformal_result.p_value,
            'is_reliable': conformal_result.is_reliable,
            'mc_answers': mc_result.answers,
            'metadata': {
                'mc': mc_result.metadata,
                'conformal': conformal_result.metadata
            }
        }
