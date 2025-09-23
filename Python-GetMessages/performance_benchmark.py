#!/usr/bin/env python3
"""
Performance benchmark comparing math-based vs PyTorch-based cosine similarity.
"""

import time
import random
import math
import statistics
import torch
import torch.nn.functional as F
from typing import List


def cosine_similarity_math(v1: list, v2: list) -> float:
    """Original math-based cosine similarity implementation."""
    if not v1 or not v2:
        return 0.0
    
    def dot(a, b):
        return float(sum(x * y for x, y in zip(a, b)))
    
    def norm(a):
        return math.sqrt(float(sum(x * x for x in a)))
    
    den = norm(v1) * norm(v2)
    if den == 0:
        return 0.0
    return dot(v1, v2) / den


def cosine_similarity_pytorch(v1: list, v2: list) -> float:
    """PyTorch-based cosine similarity implementation."""
    if not v1 or not v2:
        return 0.0
    
    tensor1 = torch.tensor(v1, dtype=torch.float32)
    tensor2 = torch.tensor(v2, dtype=torch.float32)
    
    similarity = F.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0), dim=1)
    return float(similarity.item())


def rank_by_cosine_math(reference_vec: list, candidate_vecs: list[list], candidate_texts: list[str]):
    """Original math-based ranking implementation."""
    items = []
    for i, (vec, text) in enumerate(zip(candidate_vecs, candidate_texts)):
        sim = cosine_similarity_math(reference_vec, vec)
        items.append({
            "index": i,
            "message": text,
            "cosine_similarity": sim,
        })
    items.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    return items


def rank_by_cosine_pytorch_batch(reference_vec: list, candidate_vecs: list[list], candidate_texts: list[str]):
    """PyTorch batch processing implementation."""
    if not candidate_vecs or not candidate_texts:
        return []
    
    ref_tensor = torch.tensor(reference_vec, dtype=torch.float32).unsqueeze(0)
    candidate_tensor = torch.stack([torch.tensor(vec, dtype=torch.float32) for vec in candidate_vecs])
    
    similarities = F.cosine_similarity(ref_tensor, candidate_tensor, dim=1)
    
    items = []
    for i, (sim, text) in enumerate(zip(similarities.tolist(), candidate_texts)):
        items.append({
            "index": i,
            "message": text,
            "cosine_similarity": sim,
        })
    
    items.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    return items


def rank_by_cosine_pytorch_loop(reference_vec: list, candidate_vecs: list[list], candidate_texts: list[str]):
    """PyTorch loop-based implementation (for comparison)."""
    items = []
    for i, (vec, text) in enumerate(zip(candidate_vecs, candidate_texts)):
        sim = cosine_similarity_pytorch(reference_vec, vec)
        items.append({
            "index": i,
            "message": text,
            "cosine_similarity": sim,
        })
    items.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    return items


def generate_test_data(num_vectors: int, vector_dim: int):
    """Generate random test vectors."""
    random.seed(42)  # For reproducible results
    vectors = []
    texts = []
    
    for i in range(num_vectors):
        vec = [random.uniform(-1, 1) for _ in range(vector_dim)]
        vectors.append(vec)
        texts.append(f"Message {i}")
    
    reference = [random.uniform(-1, 1) for _ in range(vector_dim)]
    return reference, vectors, texts


def benchmark_function(func, *args, num_runs=10):
    """Benchmark a function and return timing statistics."""
    times = []
    
    # Warm-up run
    func(*args)
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        result = func(*args)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'min': min(times),
        'max': max(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0,
        'result_sample': result[:3] if isinstance(result, list) and len(result) > 0 else result
    }


def run_single_similarity_benchmark():
    """Benchmark single cosine similarity calculations."""
    print("=" * 80)
    print("SINGLE COSINE SIMILARITY BENCHMARK")
    print("=" * 80)
    
    # Test with different vector dimensions
    dimensions = [100, 512, 1536]  # 1536 is Titan embedding dimension
    num_pairs = 1000
    
    for dim in dimensions:
        print(f"\nVector Dimension: {dim}")
        print("-" * 40)
        
        # Generate test vectors
        test_pairs = []
        random.seed(42)
        for _ in range(num_pairs):
            v1 = [random.uniform(-1, 1) for _ in range(dim)]
            v2 = [random.uniform(-1, 1) for _ in range(dim)]
            test_pairs.append((v1, v2))
        
        # Benchmark math implementation
        def test_math():
            return [cosine_similarity_math(v1, v2) for v1, v2 in test_pairs]
        
        # Benchmark PyTorch implementation
        def test_pytorch():
            return [cosine_similarity_pytorch(v1, v2) for v1, v2 in test_pairs]
        
        math_stats = benchmark_function(test_math, num_runs=5)
        pytorch_stats = benchmark_function(test_pytorch, num_runs=5)
        
        print(f"Math-based:    {math_stats['mean']*1000:.2f} ± {math_stats['std']*1000:.2f} ms")
        print(f"PyTorch:       {pytorch_stats['mean']*1000:.2f} ± {pytorch_stats['std']*1000:.2f} ms")
        
        speedup = math_stats['mean'] / pytorch_stats['mean']
        winner = "PyTorch" if speedup > 1 else "Math"
        print(f"Winner: {winner} ({speedup:.2f}x {'faster' if speedup > 1 else 'slower'})")


def run_ranking_benchmark():
    """Benchmark ranking/batch operations."""
    print("\n" + "=" * 80)
    print("RANKING/BATCH OPERATIONS BENCHMARK")
    print("=" * 80)
    
    # Test with different numbers of candidates
    test_sizes = [10, 50, 100, 500]
    vector_dim = 1536  # Titan embedding dimension
    
    for num_candidates in test_sizes:
        print(f"\nNumber of candidates: {num_candidates}")
        print("-" * 40)
        
        reference, candidates, texts = generate_test_data(num_candidates, vector_dim)
        
        # Benchmark different implementations
        math_stats = benchmark_function(rank_by_cosine_math, reference, candidates, texts, num_runs=5)
        pytorch_loop_stats = benchmark_function(rank_by_cosine_pytorch_loop, reference, candidates, texts, num_runs=5)
        pytorch_batch_stats = benchmark_function(rank_by_cosine_pytorch_batch, reference, candidates, texts, num_runs=5)
        
        print(f"Math (loop):        {math_stats['mean']*1000:.2f} ± {math_stats['std']*1000:.2f} ms")
        print(f"PyTorch (loop):     {pytorch_loop_stats['mean']*1000:.2f} ± {pytorch_loop_stats['std']*1000:.2f} ms")
        print(f"PyTorch (batch):    {pytorch_batch_stats['mean']*1000:.2f} ± {pytorch_batch_stats['std']*1000:.2f} ms")
        
        # Calculate speedups
        best_time = min(math_stats['mean'], pytorch_loop_stats['mean'], pytorch_batch_stats['mean'])
        
        if best_time == math_stats['mean']:
            winner = "Math (loop)"
        elif best_time == pytorch_loop_stats['mean']:
            winner = "PyTorch (loop)"
        else:
            winner = "PyTorch (batch)"
        
        batch_vs_math = math_stats['mean'] / pytorch_batch_stats['mean']
        batch_vs_loop = pytorch_loop_stats['mean'] / pytorch_batch_stats['mean']
        
        print(f"Winner: {winner}")
        print(f"PyTorch batch vs Math: {batch_vs_math:.2f}x faster")
        print(f"PyTorch batch vs PyTorch loop: {batch_vs_loop:.2f}x faster")


def main():
    print("Performance Benchmark: Math vs PyTorch Cosine Similarity")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    run_single_similarity_benchmark()
    run_ranking_benchmark()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("• For single cosine similarity calculations, the performance difference")
    print("  depends on vector size and overhead from tensor creation")
    print("• For batch operations (ranking), PyTorch batch processing shows")
    print("  significant speedups due to vectorized operations")
    print("• PyTorch is most beneficial when processing many vectors at once")


if __name__ == "__main__":
    main()
