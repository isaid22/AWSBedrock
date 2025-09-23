#!/usr/bin/env python3
"""
Show the actual tensor shapes in your embedding_work.py code
"""

import torch
import torch.nn.functional as F


def show_real_embedding_shapes():
    """Demonstrate with realistic embedding dimensions."""
    print("REAL EMBEDDING SCENARIO (like in your code)")
    print("=" * 60)
    
    # Simulate your actual data dimensions
    embedding_dim = 1536  # Titan embedding dimension
    num_candidates = 10   # Number of candidate messages
    
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of candidates: {num_candidates}")
    print()
    
    # Create mock data with realistic shapes
    import random
    random.seed(42)
    
    reference_vec = [random.uniform(-1, 1) for _ in range(embedding_dim)]
    candidate_vecs = [[random.uniform(-1, 1) for _ in range(embedding_dim)] for _ in range(num_candidates)]
    
    # Your actual tensor creation code:
    ref_tensor = torch.tensor(reference_vec, dtype=torch.float32).unsqueeze(0)
    candidate_tensor = torch.stack([torch.tensor(vec, dtype=torch.float32) for vec in candidate_vecs])
    
    print("Tensor shapes in your code:")
    print(f"ref_tensor.shape: {ref_tensor.shape}")
    print(f"candidate_tensor.shape: {candidate_tensor.shape}")
    print()
    
    print("What this means:")
    print(f"ref_tensor.shape = (1, {embedding_dim})")
    print("  - 1 reference vector")
    print(f"  - {embedding_dim} embedding features per vector")
    print()
    print(f"candidate_tensor.shape = ({num_candidates}, {embedding_dim})")
    print(f"  - {num_candidates} candidate vectors")
    print(f"  - {embedding_dim} embedding features per vector")
    print()
    
    # Your actual similarity calculation:
    similarities = F.cosine_similarity(ref_tensor, candidate_tensor, dim=1)
    
    print("Result of F.cosine_similarity(ref_tensor, candidate_tensor, dim=1):")
    print(f"similarities.shape: {similarities.shape}")
    print(f"similarities values: {similarities[:5]}...")  # Show first 5
    print()
    
    print("Why dim=1 is correct:")
    print("• dim=1 tells PyTorch to compute similarity along the embedding features")
    print("• Each similarity compares all 1536 features between reference and one candidate")
    print("• Result: one similarity score per candidate (shape: [10])")
    print("• This is exactly what you need for ranking!")
    print()
    
    print("What would happen with dim=0:")
    try:
        wrong_similarities = F.cosine_similarity(ref_tensor, candidate_tensor, dim=0)
        print(f"dim=0 result shape: {wrong_similarities.shape}")
        print("• This would give you 1536 similarity scores (one per feature)")
        print("• You'd lose the per-candidate information you need for ranking")
    except Exception as e:
        print(f"dim=0 would cause an error: {e}")


def explain_unsqueeze():
    """Explain why we use unsqueeze(0) on the reference tensor."""
    print("\n" + "=" * 60)
    print("WHY WE USE .unsqueeze(0) ON THE REFERENCE TENSOR")
    print("=" * 60)
    
    # Example with smaller dimensions
    reference_list = [0.1, 0.2, 0.3]
    
    # Without unsqueeze
    ref_tensor_1d = torch.tensor(reference_list, dtype=torch.float32)
    print(f"Without unsqueeze: ref_tensor.shape = {ref_tensor_1d.shape}")
    
    # With unsqueeze(0)
    ref_tensor_2d = torch.tensor(reference_list, dtype=torch.float32).unsqueeze(0)
    print(f"With unsqueeze(0): ref_tensor.shape = {ref_tensor_2d.shape}")
    print()
    
    print("Why we need unsqueeze(0):")
    print("• F.cosine_similarity expects both tensors to have the same number of dimensions")
    print("• candidate_tensor has shape (num_candidates, embedding_dim) - 2D")
    print("• Without unsqueeze, ref_tensor would be 1D: (embedding_dim,)")
    print("• With unsqueeze(0), ref_tensor becomes 2D: (1, embedding_dim)")
    print("• Now both tensors are 2D and can be compared element-wise")
    print()
    print("Broadcasting behavior:")
    print("• PyTorch broadcasts the (1, embedding_dim) reference")
    print("• Against each of the (num_candidates, embedding_dim) candidates")
    print("• Resulting in (num_candidates,) similarity scores")


if __name__ == "__main__":
    show_real_embedding_shapes()
    explain_unsqueeze()
