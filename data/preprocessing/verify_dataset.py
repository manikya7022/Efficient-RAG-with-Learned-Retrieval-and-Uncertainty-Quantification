"""
Dataset Verification

Validates dataset integrity, computes statistics, and generates reports.

Usage:
    python verify_dataset.py --dataset data/preprocessed/wikipedia_100k.jsonl
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_jsonl_dataset(filepath: str) -> Dict[str, Any]:
    """
    Verify JSONL dataset and compute statistics.
    
    Args:
        filepath: Path to JSONL file
        
    Returns:
        Dictionary with statistics and validation results
    """
    path = Path(filepath)
    
    if not path.exists():
        return {'error': f'File not found: {filepath}', 'valid': False}
    
    stats = {
        'filepath': str(path.absolute()),
        'file_size_mb': path.stat().st_size / (1024 * 1024),
        'file_hash': compute_file_hash(filepath),
        'valid': True,
        'errors': [],
        'document_count': 0,
        'fields': Counter(),
        'text_lengths': [],
        'id_duplicates': [],
        'sample_documents': []
    }
    
    ids_seen = set()
    
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            try:
                doc = json.loads(line.strip())
                stats['document_count'] += 1
                
                # Track fields
                for field in doc.keys():
                    stats['fields'][field] += 1
                
                # Check required fields
                if 'id' not in doc:
                    stats['errors'].append(f"Line {line_num}: Missing 'id' field")
                else:
                    if doc['id'] in ids_seen:
                        stats['id_duplicates'].append(doc['id'])
                    ids_seen.add(doc['id'])
                
                if 'text' not in doc:
                    stats['errors'].append(f"Line {line_num}: Missing 'text' field")
                else:
                    stats['text_lengths'].append(len(doc['text']))
                
                # Sample first 3 documents
                if stats['document_count'] <= 3:
                    sample = {k: v for k, v in doc.items()}
                    if 'text' in sample:
                        sample['text'] = sample['text'][:200] + '...'
                    stats['sample_documents'].append(sample)
                    
            except json.JSONDecodeError as e:
                stats['errors'].append(f"Line {line_num}: Invalid JSON - {e}")
    
    # Compute text length statistics
    if stats['text_lengths']:
        import numpy as np
        lengths = np.array(stats['text_lengths'])
        stats['text_stats'] = {
            'min': int(lengths.min()),
            'max': int(lengths.max()),
            'mean': float(lengths.mean()),
            'median': float(np.median(lengths)),
            'std': float(lengths.std())
        }
    
    # Convert Counter to dict
    stats['fields'] = dict(stats['fields'])
    
    # Determine validity
    if stats['errors']:
        stats['valid'] = False
    if stats['id_duplicates']:
        stats['valid'] = False
        stats['duplicate_count'] = len(stats['id_duplicates'])
        stats['id_duplicates'] = stats['id_duplicates'][:10]  # Show first 10
    
    # Remove large lists for display
    del stats['text_lengths']
    
    return stats


def verify_nq_dataset(filepath: str) -> Dict[str, Any]:
    """
    Verify Natural Questions dataset format.
    
    Args:
        filepath: Path to NQ JSONL file
        
    Returns:
        Dictionary with statistics and validation results
    """
    base_stats = verify_jsonl_dataset(filepath)
    
    if not base_stats['valid']:
        return base_stats
    
    # Additional NQ-specific checks
    nq_stats = {
        'questions_with_answers': 0,
        'avg_answers_per_question': 0,
        'avg_context_length': 0
    }
    
    answer_counts = []
    context_lengths = []
    
    with open(filepath) as f:
        for line in f:
            doc = json.loads(line.strip())
            
            if 'question' in doc:
                nq_stats['questions_with_answers'] += 1
            
            if 'answers' in doc:
                answer_counts.append(len(doc['answers']))
            
            if 'context' in doc:
                context_lengths.append(len(doc['context']))
    
    if answer_counts:
        import numpy as np
        nq_stats['avg_answers_per_question'] = float(np.mean(answer_counts))
    
    if context_lengths:
        import numpy as np
        nq_stats['avg_context_length'] = float(np.mean(context_lengths))
    
    base_stats.update(nq_stats)
    return base_stats


def generate_report(stats: Dict[str, Any]) -> str:
    """Generate human-readable verification report."""
    lines = [
        "=" * 60,
        "Dataset Verification Report",
        "=" * 60,
        "",
        f"File: {stats.get('filepath', 'N/A')}",
        f"Size: {stats.get('file_size_mb', 0):.2f} MB",
        f"SHA-256: {stats.get('file_hash', 'N/A')[:16]}...",
        "",
        f"Valid: {'✓ Yes' if stats.get('valid') else '✗ No'}",
        f"Documents: {stats.get('document_count', 0):,}",
        "",
        "Fields:",
    ]
    
    for field, count in stats.get('fields', {}).items():
        lines.append(f"  - {field}: {count:,}")
    
    if 'text_stats' in stats:
        lines.extend([
            "",
            "Text Length Statistics:",
            f"  - Min: {stats['text_stats']['min']:,} chars",
            f"  - Max: {stats['text_stats']['max']:,} chars",
            f"  - Mean: {stats['text_stats']['mean']:.1f} chars",
            f"  - Median: {stats['text_stats']['median']:.1f} chars",
        ])
    
    if stats.get('errors'):
        lines.extend([
            "",
            f"Errors ({len(stats['errors'])}):",
        ])
        for error in stats['errors'][:5]:
            lines.append(f"  - {error}")
        if len(stats['errors']) > 5:
            lines.append(f"  ... and {len(stats['errors']) - 5} more")
    
    if stats.get('id_duplicates'):
        lines.extend([
            "",
            f"Duplicate IDs: {stats.get('duplicate_count', len(stats['id_duplicates']))}",
        ])
    
    lines.extend([
        "",
        "Sample Documents:",
    ])
    for i, sample in enumerate(stats.get('sample_documents', []), 1):
        lines.append(f"  {i}. {sample}")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify dataset integrity")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--type", choices=["corpus", "nq"], default="corpus",
                       help="Dataset type")
    parser.add_argument("--output", help="Output JSON path for stats")
    
    args = parser.parse_args()
    
    if args.type == "nq":
        stats = verify_nq_dataset(args.dataset)
    else:
        stats = verify_jsonl_dataset(args.dataset)
    
    # Print report
    print(generate_report(stats))
    
    # Save JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {args.output}")
