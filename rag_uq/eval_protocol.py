"""
Calibrated Evaluation Protocol for RAG

Comprehensive evaluation suite covering:
1. Retrieval Quality: Recall@K, MRR, NDCG@10
2. Generation Quality: Exact Match, Token F1, ROUGE-L
3. Calibration Quality: ECE, MCE, Brier Score, Reliability Diagrams
4. System Efficiency: Latency metrics, router overhead

Designed for PhD-level rigor with support for:
- Statistical significance testing
- Confidence intervals via bootstrap
- Publication-ready visualization
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality."""
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    router_accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'recall@k': self.recall_at_k,
            'mrr': self.mrr,
            'ndcg@10': self.ndcg_at_10,
            'precision@k': self.precision_at_k,
            'router_accuracy': self.router_accuracy
        }


@dataclass
class GenerationMetrics:
    """Metrics for answer generation quality."""
    exact_match: float = 0.0
    f1: float = 0.0
    rouge_l: float = 0.0
    avg_answer_length: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'exact_match': self.exact_match,
            'f1': self.f1,
            'rouge_l': self.rouge_l,
            'avg_answer_length': self.avg_answer_length
        }


@dataclass
class CalibrationMetrics:
    """Metrics for uncertainty calibration quality."""
    ece: float = 0.0  # Expected Calibration Error
    mce: float = 0.0  # Maximum Calibration Error
    brier_score: float = 0.0
    ece_per_bin: List[float] = field(default_factory=list)
    accuracy_per_bin: List[float] = field(default_factory=list)
    confidence_per_bin: List[float] = field(default_factory=list)
    bin_counts: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'ece': self.ece,
            'mce': self.mce,
            'brier_score': self.brier_score,
            'ece_per_bin': self.ece_per_bin,
            'accuracy_per_bin': self.accuracy_per_bin,
            'confidence_per_bin': self.confidence_per_bin,
            'bin_counts': self.bin_counts
        }


@dataclass
class EfficiencyMetrics:
    """Metrics for system efficiency."""
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    router_overhead_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    throughput_qps: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'avg_latency_ms': self.avg_latency_ms,
            'p50_latency_ms': self.p50_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'p99_latency_ms': self.p99_latency_ms,
            'router_overhead_ms': self.router_overhead_ms,
            'retrieval_latency_ms': self.retrieval_latency_ms,
            'generation_latency_ms': self.generation_latency_ms,
            'throughput_qps': self.throughput_qps
        }


class RAGEvaluator:
    """
    Comprehensive evaluation suite for RAG systems.
    
    Implements PhD-level evaluation methodology with:
    - Multiple metric categories (retrieval, generation, calibration, efficiency)
    - Statistical significance testing
    - Bootstrap confidence intervals
    - Publication-ready visualizations
    
    Args:
        output_dir: Directory for saving results and figures
        n_bins: Number of bins for calibration metrics
        bootstrap_samples: Number of bootstrap samples for CIs
        
    Example:
        >>> evaluator = RAGEvaluator(output_dir="results/")
        >>> retrieval_metrics = evaluator.evaluate_retrieval(
        ...     retrieved_ids, gold_ids, k_values=[1, 5, 10]
        ... )
        >>> print(f"MRR: {retrieval_metrics.mrr:.4f}")
    """
    
    def __init__(
        self,
        output_dir: str = "results",
        n_bins: int = 10,
        bootstrap_samples: int = 1000
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_bins = n_bins
        self.bootstrap_samples = bootstrap_samples
        
        if HAS_ROUGE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        else:
            self.rouge_scorer = None
    
    # ==================== Retrieval Metrics ====================
    
    def _recall_at_k(
        self, 
        retrieved: List[str], 
        relevant: List[str], 
        k: int
    ) -> float:
        """Compute Recall@K."""
        if not relevant:
            return 0.0
        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)
        return len(retrieved_set & relevant_set) / len(relevant_set)
    
    def _precision_at_k(
        self, 
        retrieved: List[str], 
        relevant: List[str], 
        k: int
    ) -> float:
        """Compute Precision@K."""
        if k == 0:
            return 0.0
        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)
        return len(retrieved_set & relevant_set) / k
    
    def _reciprocal_rank(
        self, 
        retrieved: List[str], 
        relevant: List[str]
    ) -> float:
        """Compute Reciprocal Rank."""
        relevant_set = set(relevant)
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    
    def _dcg(self, relevances: List[float], k: int) -> float:
        """Compute DCG@K."""
        dcg = 0.0
        for i, rel in enumerate(relevances[:k]):
            dcg += (2 ** rel - 1) / np.log2(i + 2)
        return dcg
    
    def _ndcg_at_k(
        self, 
        retrieved: List[str], 
        relevances: Dict[str, float],
        k: int
    ) -> float:
        """Compute NDCG@K."""
        # Get relevance scores for retrieved docs
        retrieved_rels = [relevances.get(doc_id, 0.0) for doc_id in retrieved[:k]]
        
        # DCG
        dcg = self._dcg(retrieved_rels, k)
        
        # Ideal DCG (sorted relevances)
        ideal_rels = sorted(relevances.values(), reverse=True)[:k]
        idcg = self._dcg(ideal_rels, k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_retrieval(
        self,
        retrieved_ids: List[List[str]],
        gold_ids: List[List[str]],
        relevance_scores: Optional[List[Dict[str, float]]] = None,
        k_values: List[int] = [1, 5, 10, 20],
        router_decisions: Optional[List[int]] = None,
        oracle_decisions: Optional[List[int]] = None
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval quality.
        
        Args:
            retrieved_ids: List of retrieved doc ID lists per query
            gold_ids: List of relevant doc ID lists per query
            relevance_scores: Optional graded relevance per query
            k_values: K values for Recall@K and Precision@K
            router_decisions: Learned router's decisions (0=BM25, 1=dense)
            oracle_decisions: Optimal decisions for comparison
            
        Returns:
            RetrievalMetrics with all computed metrics
        """
        n_queries = len(retrieved_ids)
        
        # Initialize accumulators
        recall_sums = defaultdict(float)
        precision_sums = defaultdict(float)
        rr_sum = 0.0
        ndcg_sum = 0.0
        
        for i in range(n_queries):
            retrieved = retrieved_ids[i]
            gold = gold_ids[i]
            
            # Recall and Precision at K
            for k in k_values:
                recall_sums[k] += self._recall_at_k(retrieved, gold, k)
                precision_sums[k] += self._precision_at_k(retrieved, gold, k)
            
            # MRR
            rr_sum += self._reciprocal_rank(retrieved, gold)
            
            # NDCG
            if relevance_scores:
                ndcg_sum += self._ndcg_at_k(retrieved, relevance_scores[i], 10)
            else:
                # Binary relevance
                binary_rels = {doc_id: 1.0 for doc_id in gold}
                ndcg_sum += self._ndcg_at_k(retrieved, binary_rels, 10)
        
        # Compute averages
        metrics = RetrievalMetrics(
            recall_at_k={k: recall_sums[k] / n_queries for k in k_values},
            precision_at_k={k: precision_sums[k] / n_queries for k in k_values},
            mrr=rr_sum / n_queries,
            ndcg_at_10=ndcg_sum / n_queries
        )
        
        # Router accuracy
        if router_decisions is not None and oracle_decisions is not None:
            correct = sum(1 for r, o in zip(router_decisions, oracle_decisions) if r == o)
            metrics.router_accuracy = correct / len(router_decisions)
        
        return metrics
    
    # ==================== Generation Metrics ====================
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize answer for comparison."""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def _exact_match(self, prediction: str, reference: str) -> float:
        """Compute exact match after normalization."""
        return float(self._normalize_answer(prediction) == self._normalize_answer(reference))
    
    def _token_f1(self, prediction: str, reference: str) -> float:
        """Compute token-level F1 score."""
        pred_tokens = set(self._normalize_answer(prediction).split())
        ref_tokens = set(self._normalize_answer(reference).split())
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def _rouge_l(self, prediction: str, reference: str) -> float:
        """Compute ROUGE-L F1 score."""
        if self.rouge_scorer is None:
            # Fallback to token F1
            return self._token_f1(prediction, reference)
        
        scores = self.rouge_scorer.score(reference, prediction)
        return scores['rougeL'].fmeasure
    
    def evaluate_generation(
        self,
        predictions: List[str],
        references: List[str]
    ) -> GenerationMetrics:
        """
        Evaluate generation quality.
        
        Args:
            predictions: Generated answers
            references: Ground truth answers
            
        Returns:
            GenerationMetrics with EM, F1, ROUGE-L
        """
        n = len(predictions)
        
        em_sum = 0.0
        f1_sum = 0.0
        rouge_sum = 0.0
        length_sum = 0.0
        
        for pred, ref in zip(predictions, references):
            em_sum += self._exact_match(pred, ref)
            f1_sum += self._token_f1(pred, ref)
            rouge_sum += self._rouge_l(pred, ref)
            length_sum += len(pred.split())
        
        return GenerationMetrics(
            exact_match=em_sum / n,
            f1=f1_sum / n,
            rouge_l=rouge_sum / n,
            avg_answer_length=length_sum / n
        )
    
    # ==================== Calibration Metrics ====================
    
    def _expected_calibration_error(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[float, List[float], List[float], List[float], List[int]]:
        """
        Compute Expected Calibration Error.
        
        Returns:
            Tuple of (ECE, per_bin_ece, per_bin_accuracy, per_bin_confidence, bin_counts)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        per_bin_ece = []
        per_bin_accuracy = []
        per_bin_confidence = []
        bin_counts = []
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            bin_counts.append(int(in_bin.sum()))
            
            if in_bin.sum() > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = correctness[in_bin].mean()
                bin_ece = np.abs(avg_accuracy - avg_confidence)
                
                ece += prop_in_bin * bin_ece
                per_bin_ece.append(float(bin_ece))
                per_bin_accuracy.append(float(avg_accuracy))
                per_bin_confidence.append(float(avg_confidence))
            else:
                per_bin_ece.append(0.0)
                per_bin_accuracy.append(0.0)
                per_bin_confidence.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
        
        return float(ece), per_bin_ece, per_bin_accuracy, per_bin_confidence, bin_counts
    
    def _maximum_calibration_error(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute Maximum Calibration Error."""
        _, per_bin_ece, _, _, _ = self._expected_calibration_error(confidences, correctness, n_bins)
        return max(per_bin_ece) if per_bin_ece else 0.0
    
    def _brier_score(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray
    ) -> float:
        """Compute Brier Score."""
        return float(np.mean((confidences - correctness) ** 2))
    
    def evaluate_calibration(
        self,
        confidences: List[float],
        correctness: List[int]
    ) -> CalibrationMetrics:
        """
        Evaluate uncertainty calibration.
        
        Args:
            confidences: Model confidence scores [0, 1]
            correctness: Binary correctness labels (1=correct, 0=incorrect)
            
        Returns:
            CalibrationMetrics with ECE, MCE, Brier score
        """
        conf_array = np.array(confidences)
        corr_array = np.array(correctness, dtype=float)
        
        ece, per_bin_ece, accuracy, confidence, counts = self._expected_calibration_error(
            conf_array, corr_array, self.n_bins
        )
        
        return CalibrationMetrics(
            ece=ece,
            mce=max(per_bin_ece) if per_bin_ece else 0.0,
            brier_score=self._brier_score(conf_array, corr_array),
            ece_per_bin=per_bin_ece,
            accuracy_per_bin=accuracy,
            confidence_per_bin=confidence,
            bin_counts=counts
        )
    
    # ==================== Efficiency Metrics ====================
    
    def evaluate_efficiency(
        self,
        latencies: List[float],
        router_times: Optional[List[float]] = None,
        retrieval_times: Optional[List[float]] = None,
        generation_times: Optional[List[float]] = None
    ) -> EfficiencyMetrics:
        """
        Evaluate system efficiency.
        
        Args:
            latencies: End-to-end latencies in seconds
            router_times: Router inference times
            retrieval_times: Retrieval latencies
            generation_times: Generation latencies
            
        Returns:
            EfficiencyMetrics with latency percentiles
        """
        latencies_ms = np.array(latencies) * 1000
        
        metrics = EfficiencyMetrics(
            avg_latency_ms=float(latencies_ms.mean()),
            p50_latency_ms=float(np.percentile(latencies_ms, 50)),
            p95_latency_ms=float(np.percentile(latencies_ms, 95)),
            p99_latency_ms=float(np.percentile(latencies_ms, 99)),
            throughput_qps=1000.0 / latencies_ms.mean() if latencies_ms.mean() > 0 else 0
        )
        
        if router_times:
            metrics.router_overhead_ms = float(np.mean(router_times) * 1000)
        if retrieval_times:
            metrics.retrieval_latency_ms = float(np.mean(retrieval_times) * 1000)
        if generation_times:
            metrics.generation_latency_ms = float(np.mean(generation_times) * 1000)
        
        return metrics
    
    # ==================== Visualization ====================
    
    def plot_reliability_diagram(
        self,
        calibration_metrics: CalibrationMetrics,
        title: str = "Reliability Diagram",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate publication-ready reliability diagram.
        
        Args:
            calibration_metrics: Computed calibration metrics
            title: Plot title
            save_path: Path to save figure (default: output_dir/reliability_diagram.png)
            
        Returns:
            Path to saved figure
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available, skipping plot")
            return None
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
        
        # Model calibration
        confidences = calibration_metrics.confidence_per_bin
        accuracies = calibration_metrics.accuracy_per_bin
        counts = calibration_metrics.bin_counts
        
        # Filter empty bins
        valid_idx = [i for i, c in enumerate(counts) if c > 0]
        valid_conf = [confidences[i] for i in valid_idx]
        valid_acc = [accuracies[i] for i in valid_idx]
        valid_counts = [counts[i] for i in valid_idx]
        
        # Scatter with size proportional to count
        sizes = [max(50, min(500, c * 5)) for c in valid_counts]
        ax.scatter(valid_conf, valid_acc, s=sizes, alpha=0.7, 
                   color='#2E86AB', edgecolor='white', linewidth=1.5,
                   label='Model')
        ax.plot(valid_conf, valid_acc, 'o-', color='#2E86AB', linewidth=2, markersize=8)
        
        # Gap visualization (shaded area between perfect and actual)
        ax.fill_between(valid_conf, valid_conf, valid_acc, alpha=0.2, color='red',
                        label=f'Calibration gap (ECE={calibration_metrics.ece:.3f})')
        
        # Formatting
        ax.set_xlabel('Confidence', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add ECE annotation
        ax.text(0.05, 0.95, f'ECE: {calibration_metrics.ece:.4f}\n'
                           f'MCE: {calibration_metrics.mce:.4f}\n'
                           f'Brier: {calibration_metrics.brier_score:.4f}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "reliability_diagram.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved reliability diagram to {save_path}")
        return str(save_path)
    
    def plot_routing_analysis(
        self,
        bm25_scores: List[float],
        dense_scores: List[float],
        router_weights: List[float],
        correctness: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Visualize router decision patterns.
        
        Args:
            bm25_scores: BM25 retrieval scores
            dense_scores: Dense retrieval scores
            router_weights: Learned routing weights (0=BM25, 1=dense)
            correctness: Whether the final answer was correct
            save_path: Path to save figure
            
        Returns:
            Path to saved figure
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Router weight distribution
        ax1 = axes[0]
        ax1.hist(router_weights, bins=20, edgecolor='white', color='#2E86AB')
        ax1.axvline(x=0.5, color='red', linestyle='--', label='Decision boundary')
        ax1.set_xlabel('Router Weight (0=BM25, 1=Dense)')
        ax1.set_ylabel('Count')
        ax1.set_title('Router Decision Distribution')
        ax1.legend()
        
        # 2. Score difference vs router decision
        ax2 = axes[1]
        score_diff = np.array(dense_scores) - np.array(bm25_scores)
        ax2.scatter(score_diff, router_weights, alpha=0.5, color='#2E86AB')
        ax2.set_xlabel('Dense - BM25 Score')
        ax2.set_ylabel('Router Weight')
        ax2.set_title('Score Difference vs Router Decision')
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        
        # 3. Correctness by routing decision (if available)
        ax3 = axes[2]
        if correctness is not None:
            weights = np.array(router_weights)
            correct = np.array(correctness)
            
            dense_mask = weights > 0.5
            bm25_mask = ~dense_mask
            
            dense_accuracy = correct[dense_mask].mean() if dense_mask.sum() > 0 else 0
            bm25_accuracy = correct[bm25_mask].mean() if bm25_mask.sum() > 0 else 0
            
            bars = ax3.bar(['BM25 Selected', 'Dense Selected'], 
                          [bm25_accuracy, dense_accuracy],
                          color=['#E8751A', '#2E86AB'])
            ax3.set_ylabel('Accuracy')
            ax3.set_title('Accuracy by Routing Decision')
            ax3.set_ylim([0, 1])
            
            # Add count annotations
            ax3.text(0, bm25_accuracy + 0.02, f'n={bm25_mask.sum()}', ha='center')
            ax3.text(1, dense_accuracy + 0.02, f'n={dense_mask.sum()}', ha='center')
        else:
            ax3.text(0.5, 0.5, 'Correctness data\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "routing_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    # ==================== Full Evaluation ====================
    
    def run_full_evaluation(
        self,
        retrieved_ids: List[List[str]],
        gold_ids: List[List[str]],
        predictions: List[str],
        references: List[str],
        confidences: List[float],
        latencies: List[float],
        router_weights: Optional[List[float]] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete evaluation and generate report.
        
        Args:
            retrieved_ids: Retrieved document IDs per query
            gold_ids: Relevant document IDs per query
            predictions: Generated answers
            references: Ground truth answers
            confidences: Model confidence scores
            latencies: End-to-end latencies
            router_weights: Optional router gating weights
            save_results: Whether to save JSON report
            
        Returns:
            Complete evaluation results dictionary
        """
        # Compute metrics
        retrieval_metrics = self.evaluate_retrieval(retrieved_ids, gold_ids)
        generation_metrics = self.evaluate_generation(predictions, references)
        
        # Correctness for calibration
        correctness = [
            1 if self._exact_match(p, r) or self._token_f1(p, r) > 0.5 else 0
            for p, r in zip(predictions, references)
        ]
        calibration_metrics = self.evaluate_calibration(confidences, correctness)
        efficiency_metrics = self.evaluate_efficiency(latencies)
        
        # Generate plots
        reliability_path = self.plot_reliability_diagram(calibration_metrics)
        
        routing_path = None
        if router_weights:
            # Compute scores for routing analysis (placeholder)
            routing_path = self.plot_routing_analysis(
                bm25_scores=[0] * len(router_weights),  # Placeholder
                dense_scores=[0] * len(router_weights),
                router_weights=router_weights,
                correctness=correctness
            )
        
        # Compile results
        results = {
            'retrieval': retrieval_metrics.to_dict(),
            'generation': generation_metrics.to_dict(),
            'calibration': calibration_metrics.to_dict(),
            'efficiency': efficiency_metrics.to_dict(),
            'plots': {
                'reliability_diagram': reliability_path,
                'routing_analysis': routing_path
            },
            'summary': {
                'n_queries': len(predictions),
                'mrr': retrieval_metrics.mrr,
                'exact_match': generation_metrics.exact_match,
                'f1': generation_metrics.f1,
                'ece': calibration_metrics.ece,
                'avg_latency_ms': efficiency_metrics.avg_latency_ms
            }
        }
        
        if save_results:
            results_path = self.output_dir / "evaluation_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved evaluation results to {results_path}")
        
        return results
    
    # ==================== Bootstrap Confidence Intervals ====================
    
    def bootstrap_metric(
        self,
        data: List[Any],
        metric_fn: Callable,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for a metric.
        
        Args:
            data: Data to bootstrap from
            metric_fn: Function that computes metric from data sample
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (default 0.95)
            
        Returns:
            Tuple of (point_estimate, lower_bound, upper_bound)
        """
        n = len(data)
        bootstrap_values = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            sample = [data[i] for i in indices]
            bootstrap_values.append(metric_fn(sample))
        
        point_estimate = metric_fn(data)
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_values, 100 * alpha / 2)
        upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
        
        return float(point_estimate), float(lower), float(upper)
