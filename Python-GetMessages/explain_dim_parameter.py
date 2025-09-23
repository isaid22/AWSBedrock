#!/usr/bin/env python3
"""
Demonstration of why we use dim=1 in F.cosine_similarity()
"""

import torch
import torch.nn.functional as F


def demonstrate_cosine_similarity_dimensions():
    """Show tensor shapes and explain why dim=1 is used."""
    
    print("Understanding dim=1 in F.cosine_similarity()")
    print("=" * 60)
    
    # Create example data similar to your embedding scenario
    embedding_dim = 5  # Using small dimension for clarity (real Titan embeddings are 1536)
    num_candidates = 3
    
    # Reference vector (what you're comparing against)
    reference_vec = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Candidate vectors (what you're ranking)
    candidate_vecs = [
        [0.2, 0.1, 0.4, 0.3, 0.6],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.3, 0.2, 0.5, 0.4]
    ]
    
    print(f"Reference vector shape: {len(reference_vec)}")
    print(f"Number of candidates: {num_candidates}")
    print(f"Each candidate vector shape: {len(candidate_vecs[0])}")
    print()
    
    # Convert to tensors as in your code
    ref_tensor = torch.tensor(reference_vec, dtype=torch.float32).unsqueeze(0)
    candidate_tensor = torch.stack([torch.tensor(vec, dtype=torch.float32) for vec in candidate_vecs])
    
    print("Tensor shapes after conversion:")
    print(f"ref_tensor.shape: {ref_tensor.shape}")
    print(f"candidate_tensor.shape: {candidate_tensor.shape}")
    print()
    
    print("Breaking down the shapes:")
    print("ref_tensor.shape = (1, 5)")
    print("  - Dimension 0: batch size = 1 (single reference)")
    print("  - Dimension 1: embedding features = 5")
    print()
    print("candidate_tensor.shape = (3, 5)")
    print("  - Dimension 0: batch size = 3 (three candidates)")
    print("  - Dimension 1: embedding features = 5")
    print()
    
    # Show what happens with different dim values
    print("Testing different dim values:")
    print("-" * 40)
    
    try:
        # dim=1: Compute similarity along the feature dimension
        similarities_dim1 = F.cosine_similarity(ref_tensor, candidate_tensor, dim=1)
        print(f"dim=1 result shape: {similarities_dim1.shape}")
        print(f"dim=1 result values: {similarities_dim1}")
        print("✓ This gives us one similarity score per candidate (what we want!)")
        print()
    except Exception as e:
        print(f"dim=1 error: {e}")
    
    try:
        # dim=0: Compute similarity along the batch dimension
        similarities_dim0 = F.cosine_similarity(ref_tensor, candidate_tensor, dim=0)
        print(f"dim=0 result shape: {similarities_dim0.shape}")
        print(f"dim=0 result values: {similarities_dim0}")
        print("✗ This gives us one score per feature dimension (not what we want)")
        print()
    except Exception as e:
        print(f"dim=0 error: {e}")
    
    # Manual calculation to verify dim=1 is correct
    print("Manual verification of dim=1 results:")
    print("-" * 40)
    
    def manual_cosine_similarity(v1, v2):
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(a * a for a in v2) ** 0.5
        return dot_product / (norm1 * norm2)
    
    for i, candidate in enumerate(candidate_vecs):
        manual_sim = manual_cosine_similarity(reference_vec, candidate)
        pytorch_sim = similarities_dim1[i].item()
        print(f"Candidate {i}: Manual={manual_sim:.6f}, PyTorch={pytorch_sim:.6f}")
    
    print()
    print("Key insight:")
    print("dim=1 tells PyTorch to compute cosine similarity across the")
    print("feature dimension (the actual embedding values), giving us")
    print("one similarity score between each candidate and the reference.")


def show_dimension_concept():
    """Visual explanation of tensor dimensions."""
    print("\n" + "=" * 60)
    print("VISUAL EXPLANATION OF TENSOR DIMENSIONS")
    print("=" * 60)
    
    print("""
Tensor dimensions explained:

ref_tensor shape: (1, 5)
┌─────────────────────┐
│  0.1  0.2  0.3  0.4  0.5  │ ← Reference vector (1 row, 5 columns)
└─────────────────────┘
  ↑                     ↑
  dim=0                dim=1
  (batch)              (features)

candidate_tensor shape: (3, 5)
┌─────────────────────┐
│  0.2  0.1  0.4  0.3  0.6  │ ← Candidate 0
│  0.5  0.4  0.3  0.2  0.1  │ ← Candidate 1  
│  0.1  0.3  0.2  0.5  0.4  │ ← Candidate 2
└─────────────────────┘
  ↑                     ↑
  dim=0                dim=1
  (batch)              (features)

When we use dim=1:
- PyTorch computes cosine similarity along the feature dimension
- It compares: ref[0,:] with candidate[0,:], ref[0,:] with candidate[1,:], etc.
- Result: one similarity score per candidate → shape (3,)

When we use dim=0:
- PyTorch would compute cosine similarity along the batch dimension
- It would compare feature-by-feature across batches
- Result: one score per feature → shape (5,) ← Not what we want!
""")


if __name__ == "__main__":
    demonstrate_cosine_similarity_dimensions()
    show_dimension_concept()
