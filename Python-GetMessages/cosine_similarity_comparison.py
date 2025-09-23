#!/usr/bin/env python3
"""
Comparison script to validate that PyTorch cosine similarity 
produces the same results as the math-based implementation.
"""

import math
import torch
import torch.nn.functional as F


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
    
    # Convert lists to PyTorch tensors
    tensor1 = torch.tensor(v1, dtype=torch.float32)
    tensor2 = torch.tensor(v2, dtype=torch.float32)
    
    # Use PyTorch's cosine similarity function
    similarity = F.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0), dim=1)
    return float(similarity.item())


def test_cosine_similarity_implementations():
    """Test that both implementations produce identical results."""
    
    # Test cases
    test_vectors = [
        ([1, 2, 3], [4, 5, 6]),
        ([1, 0, 0], [0, 1, 0]),  # Orthogonal vectors
        ([1, 1, 1], [1, 1, 1]),  # Identical vectors
        ([1, 2, 3], [-1, -2, -3]),  # Opposite vectors
        ([0.5, 0.8, 0.2], [0.1, 0.9, 0.4]),  # Fractional values
        ([], [1, 2, 3]),  # Empty vector
        ([1, 2, 3], []),  # Empty vector
        ([0, 0, 0], [1, 2, 3]),  # Zero vector
    ]
    
    print("Comparing cosine similarity implementations:")
    print("=" * 60)
    
    all_match = True
    tolerance = 1e-6
    
    for i, (v1, v2) in enumerate(test_vectors):
        math_result = cosine_similarity_math(v1, v2)
        pytorch_result = cosine_similarity_pytorch(v1, v2)
        
        # Check if results match within tolerance
        diff = abs(math_result - pytorch_result)
        matches = diff < tolerance
        all_match = all_match and matches
        
        status = "✓" if matches else "✗"
        print(f"Test {i+1}: {status}")
        print(f"  Vectors: {v1[:3]}{'...' if len(v1) > 3 else ''} vs {v2[:3]}{'...' if len(v2) > 3 else ''}")
        print(f"  Math result:    {math_result:.8f}")
        print(f"  PyTorch result: {pytorch_result:.8f}")
        print(f"  Difference:     {diff:.2e}")
        print()
    
    print("=" * 60)
    if all_match:
        print("✅ All tests passed! PyTorch implementation matches math implementation.")
    else:
        print("❌ Some tests failed! Results don't match.")
    
    return all_match


if __name__ == "__main__":
    test_cosine_similarity_implementations()
