"""
Full Evaluation Pipeline

Runs comprehensive evaluation of the RAG system with learned routing
and uncertainty quantification.

Usage:
    python run_evaluation.py --nq-path data/preprocessed/nq_dev_3000.jsonl
"""

import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import os

import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_uq.router import RetrievalRouter, RouterConfig
from rag_uq.streaming_index import HybridRetriever
from rag_uq.eval_protocol import RAGEvaluator

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


def load_test_samples(
    nq_path: str,
    skip_first: int = 500,
    max_samples: int = 500
) -> List[Dict[str, Any]]:
    """Load test samples, skipping calibration set."""
    samples = []
    
    with open(nq_path) as f:
        for i, line in enumerate(f):
            if i < skip_first:
                continue
            if len(samples) >= max_samples:
                break
            
            try:
                sample = json.loads(line.strip())
                if (sample.get('question') and 
                    sample.get('answers') and 
                    sample.get('context')):
                    samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    return samples


def generate_answer(
    llm_client,
    question: str,
    context: str,
    model: str = "llama3.2:3b"
) -> str:
    """Generate answer using LLM."""
    prompt = f"""Answer the following question based on the provided context.
Be concise and give only the answer.

Context: {context}

Question: {question}

Answer:"""
    
    try:
        response = llm_client.generate(
            model=model,
            prompt=prompt,
            options={'temperature': 0.1, 'num_predict': 50}
        )
        return response.get('response', '').strip()
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return ""


def run_evaluation(
    nq_path: str,
    retriever: HybridRetriever,
    router_path: Optional[str] = None,
    output_dir: str = "results",
    model: str = "llama3.2:3b",
    ollama_host: str = "http://localhost:11434",
    n_samples: int = 200,
    use_router: bool = True,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Run full evaluation pipeline.
    
    Args:
        nq_path: Path to NQ dataset
        retriever: Initialized hybrid retriever
        router_path: Path to trained router checkpoint
        output_dir: Output directory for results
        model: Ollama model name
        ollama_host: Ollama API host
        n_samples: Number of test samples
        use_router: Whether to use learned router
        debug: Enable debug logging
        
    Returns:
        Evaluation results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load router if specified
    router = None
    if use_router and router_path and Path(router_path).exists():
        checkpoint = torch.load(router_path, map_location='cpu')
        router = RetrievalRouter(checkpoint.get('config', RouterConfig()))
        router.load_state_dict(checkpoint['model_state_dict'])
        router.eval()
        logger.info(f"Loaded router from {router_path}")
    
    # Initialize LLM client
    if HAS_OLLAMA:
        llm_client = ollama.Client(host=ollama_host)
    else:
        class MockLLM:
            def generate(self, **kwargs):
                return {'response': 'mock answer for testing'}
        llm_client = MockLLM()
        logger.warning("Using mock LLM")
    
    # Load test samples
    samples = load_test_samples(nq_path, skip_first=500, max_samples=n_samples)
    logger.info(f"Loaded {len(samples)} test samples")
    
    # Run evaluation
    retrieved_ids = []
    gold_ids = []
    predictions = []
    references = []
    confidences = []
    latencies = []
    router_weights = []
    
    for i, sample in enumerate(samples):
        start_time = time.time()
        
        question = sample['question']
        true_answers = sample['answers']
        context = sample.get('context', '')[:2000]
        
        # Retrieve passages
        bm25_scores, dense_scores, doc_ids, texts = retriever.get_scores_for_router(
            question, num_passages=10
        )
        
        # Apply router if available
        if router is not None:
            with torch.no_grad():
                bm25_tensor = torch.tensor([bm25_scores], dtype=torch.float32)
                dense_tensor = torch.tensor([dense_scores], dtype=torch.float32)
                weights = router(bm25_tensor, dense_tensor)[0]
                
                # Compute hybrid scores
                hybrid_scores = weights * dense_tensor[0] + (1 - weights) * bm25_tensor[0]
                
                # Reorder by hybrid score
                sorted_indices = torch.argsort(hybrid_scores, descending=True).tolist()
                doc_ids = [doc_ids[i] for i in sorted_indices]
                texts = [texts[i] for i in sorted_indices]
                
                router_weights.append(weights.mean().item())
        else:
            router_weights.append(0.5)
        
        # Use retrieved text as context (or fall back to original context)
        retrieved_context = " ".join([t for t in texts[:3] if t]) or context
        
        # Generate answer
        prediction = generate_answer(llm_client, question, retrieved_context, model)
        
        # Compute simple confidence (placeholder - TODO: integrate MC dropout)
        # Using prediction length as proxy
        confidence = min(1.0, len(prediction.split()) / 10)
        
        latency = time.time() - start_time
        
        # Store results
        retrieved_ids.append([d for d in doc_ids if d])
        gold_ids.append([])  # NQ doesn't have gold passage IDs
        predictions.append(prediction)
        references.append(true_answers[0])
        confidences.append(confidence)
        latencies.append(latency)
        
        if debug and (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(samples)}")
            logger.info(f"  Q: {question[:50]}...")
            logger.info(f"  A: {prediction[:50]}...")
            logger.info(f"  True: {true_answers[0][:50]}...")
    
    # Compute correctness for calibration
    evaluator = RAGEvaluator(output_dir=output_dir)
    
    # Generation metrics
    gen_metrics = evaluator.evaluate_generation(predictions, references)
    
    # Calibration metrics
    correctness = [
        1 if evaluator._exact_match(p, r) or evaluator._token_f1(p, r) > 0.5 else 0
        for p, r in zip(predictions, references)
    ]
    cal_metrics = evaluator.evaluate_calibration(confidences, correctness)
    
    # Efficiency metrics
    eff_metrics = evaluator.evaluate_efficiency(latencies)
    
    # Generate plots
    reliability_path = evaluator.plot_reliability_diagram(cal_metrics)
    routing_path = evaluator.plot_routing_analysis(
        bm25_scores=[0] * len(router_weights),  # Placeholder
        dense_scores=[0] * len(router_weights),
        router_weights=router_weights,
        correctness=correctness
    )
    
    results = {
        'n_samples': len(samples),
        'generation': gen_metrics.to_dict(),
        'calibration': cal_metrics.to_dict(),
        'efficiency': eff_metrics.to_dict(),
        'router': {
            'used': router is not None,
            'avg_weight': np.mean(router_weights),
            'std_weight': np.std(router_weights)
        },
        'plots': {
            'reliability_diagram': reliability_path,
            'routing_analysis': routing_path
        },
        'summary': {
            'exact_match': gen_metrics.exact_match,
            'f1': gen_metrics.f1,
            'ece': cal_metrics.ece,
            'avg_latency_ms': eff_metrics.avg_latency_ms
        }
    }
    
    return results


def run_ablation_study(
    nq_path: str,
    retriever: HybridRetriever,
    router_path: Optional[str] = None,
    output_dir: str = "results",
    model: str = "llama3.2:3b",
    n_samples: int = 100
) -> Dict[str, Any]:
    """
    Run ablation study comparing:
    1. BM25 only
    2. Dense only
    3. Simple averaging
    4. Learned router
    """
    results = {}
    
    # 1. BM25 only (no router, use BM25 scores)
    logger.info("Running BM25-only baseline...")
    # TODO: Implement BM25-only retrieval path
    
    # 2. Dense only
    logger.info("Running Dense-only baseline...")
    # TODO: Implement Dense-only retrieval path
    
    # 3. Simple averaging (50/50)
    logger.info("Running simple averaging baseline...")
    # TODO: Implement fixed 50/50 averaging
    
    # 4. Learned router
    logger.info("Running learned router...")
    results['learned_router'] = run_evaluation(
        nq_path=nq_path,
        retriever=retriever,
        router_path=router_path,
        output_dir=f"{output_dir}/learned_router",
        model=model,
        n_samples=n_samples,
        use_router=True
    )
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--nq-path", default="data/preprocessed/nq_dev_3000.jsonl",
                       help="Path to NQ dataset")
    parser.add_argument("--chroma-path", default="./data/chroma_db",
                       help="ChromaDB path")
    parser.add_argument("--bm25-path", default="./data/bm25_index.pkl",
                       help="BM25 index path")
    parser.add_argument("--router-path", default="models/router_lora/best_router.pt",
                       help="Path to trained router")
    parser.add_argument("--output-dir", default="results",
                       help="Output directory")
    parser.add_argument("--model", default="llama3.2:3b",
                       help="Ollama model name")
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                       help="Ollama API host")
    parser.add_argument("--n-samples", type=int, default=200,
                       help="Number of test samples")
    parser.add_argument("--no-router", action="store_true",
                       help="Disable learned router")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--ablation", action="store_true",
                       help="Run ablation study")
    
    args = parser.parse_args()
    
    # Initialize retriever
    retriever = HybridRetriever(
        bm25_persist_path=args.bm25_path,
        chroma_persist_path=args.chroma_path
    )
    
    if len(retriever) == 0:
        logger.warning("Retriever is empty! Evaluation may not produce meaningful results.")
    
    if args.ablation:
        results = run_ablation_study(
            nq_path=args.nq_path,
            retriever=retriever,
            router_path=args.router_path,
            output_dir=args.output_dir,
            model=args.model,
            n_samples=args.n_samples
        )
    else:
        results = run_evaluation(
            nq_path=args.nq_path,
            retriever=retriever,
            router_path=args.router_path if not args.no_router else None,
            output_dir=args.output_dir,
            model=args.model,
            ollama_host=args.ollama_host,
            n_samples=args.n_samples,
            use_router=not args.no_router,
            debug=args.debug
        )
    
    # Save results
    results_path = Path(args.output_dir) / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nEvaluation Results:")
    print(json.dumps(results.get('summary', results), indent=2))
    logger.info(f"Full results saved to {results_path}")
