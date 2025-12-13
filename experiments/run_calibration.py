"""
Conformal Calibration Experiment

Builds the conformal prediction calibration set for uncertainty quantification.

Usage:
    python run_calibration.py --nq-path data/preprocessed/nq_dev_3000.jsonl
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    logger.warning("ollama not installed")


def load_calibration_samples(
    nq_path: str,
    n_samples: int = 500
) -> List[Dict[str, Any]]:
    """Load samples for calibration."""
    samples = []
    
    with open(nq_path) as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            
            try:
                sample = json.loads(line.strip())
                if (sample.get('question') and 
                    sample.get('answers') and 
                    sample.get('context')):
                    samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(samples)} calibration samples")
    return samples


def run_calibration(
    nq_path: str,
    output_dir: str = "data",
    n_samples: int = 500,
    model: str = "llama3.2:3b",
    ollama_host: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """
    Run conformal calibration.
    
    Args:
        nq_path: Path to NQ dataset
        output_dir: Directory for calibration database
        n_samples: Number of calibration samples
        model: Ollama model name
        ollama_host: Ollama API host
        
    Returns:
        Calibration statistics
    """
    from rag_uq.confidence import ConformalRAG
    
    # Load samples
    samples = load_calibration_samples(nq_path, n_samples)
    
    questions = [s['question'] for s in samples]
    contexts = [s['context'][:2000] for s in samples]  # Truncate long contexts
    answers = [s['answers'][0] for s in samples]  # Use first answer
    
    # Initialize conformal predictor
    if HAS_OLLAMA:
        llm_client = ollama.Client(host=ollama_host)
    else:
        # Mock client for testing
        class MockLLM:
            def generate(self, **kwargs):
                return {'response': 'mock answer'}
        llm_client = MockLLM()
        logger.warning("Using mock LLM client")
    
    calibration_db_path = f"{output_dir}/calibration_scores.db"
    conformal = ConformalRAG(
        llm_client=llm_client,
        calibration_db_path=calibration_db_path,
        alpha=0.1
    )
    
    # Run calibration
    logger.info("Starting calibration...")
    stats = conformal.calibrate(
        questions=questions,
        contexts=contexts,
        true_answers=answers,
        model=model,
        skip_existing=True
    )
    
    # Get full statistics
    full_stats = conformal.get_calibration_stats()
    stats.update(full_stats)
    
    logger.info(f"Calibration complete: {stats}")
    
    return stats


def test_conformal_predictions(
    nq_path: str,
    calibration_db_path: str = "data/calibration_scores.db",
    n_test: int = 50,
    model: str = "llama3.2:3b",
    ollama_host: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """
    Test conformal predictions on held-out samples.
    
    Args:
        nq_path: Path to NQ dataset
        calibration_db_path: Path to calibration database
        n_test: Number of test samples
        model: Ollama model name
        ollama_host: Ollama API host
        
    Returns:
        Test results
    """
    from rag_uq.confidence import ConformalRAG
    
    # Load test samples (skip calibration samples)
    samples = []
    with open(nq_path) as f:
        for i, line in enumerate(f):
            if i < 500:  # Skip calibration samples
                continue
            if len(samples) >= n_test:
                break
            
            try:
                sample = json.loads(line.strip())
                if (sample.get('question') and 
                    sample.get('answers') and 
                    sample.get('context')):
                    samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    # Initialize conformal predictor
    if HAS_OLLAMA:
        llm_client = ollama.Client(host=ollama_host)
    else:
        class MockLLM:
            def generate(self, **kwargs):
                return {'response': 'mock answer'}
        llm_client = MockLLM()
    
    conformal = ConformalRAG(
        llm_client=llm_client,
        calibration_db_path=calibration_db_path,
        alpha=0.1
    )
    
    # Test predictions
    results = []
    reliable_count = 0
    
    for sample in samples:
        result = conformal.predict_with_coverage(
            question=sample['question'],
            context=sample['context'][:2000],
            model=model
        )
        
        if result.is_reliable:
            reliable_count += 1
        
        results.append({
            'question': sample['question'],
            'prediction': result.prediction[:100],
            'true_answer': sample['answers'][0],
            'confidence': result.confidence,
            'p_value': result.p_value,
            'is_reliable': result.is_reliable
        })
    
    return {
        'n_test': len(samples),
        'reliable_ratio': reliable_count / len(samples) if samples else 0,
        'avg_confidence': sum(r['confidence'] for r in results) / len(results) if results else 0,
        'sample_results': results[:5]  # First 5 for inspection
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run conformal calibration")
    parser.add_argument("--nq-path", default="data/preprocessed/nq_dev_3000.jsonl",
                       help="Path to NQ dataset")
    parser.add_argument("--output-dir", default="data",
                       help="Output directory")
    parser.add_argument("--n-samples", type=int, default=500,
                       help="Number of calibration samples")
    parser.add_argument("--model", default="llama3.2:3b",
                       help="Ollama model name")
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                       help="Ollama API host")
    parser.add_argument("--test", action="store_true",
                       help="Run test predictions after calibration")
    
    args = parser.parse_args()
    
    # Run calibration
    cal_stats = run_calibration(
        nq_path=args.nq_path,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        model=args.model,
        ollama_host=args.ollama_host
    )
    
    print("\nCalibration Statistics:")
    print(json.dumps(cal_stats, indent=2))
    
    # Run tests if requested
    if args.test:
        test_results = test_conformal_predictions(
            nq_path=args.nq_path,
            calibration_db_path=f"{args.output_dir}/calibration_scores.db",
            model=args.model,
            ollama_host=args.ollama_host
        )
        
        print("\nTest Results:")
        print(json.dumps(test_results, indent=2))
    
    # Save results
    results_path = Path(args.output_dir) / "calibration_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'calibration': cal_stats,
            'test': test_results if args.test else None
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
