#!/usr/bin/env python3
"""
Example script showing how to test different Titan embedding dimensions.
This demonstrates the performance and quality trade-offs.
"""

from embedding_work import get_titan_embedding, get_titan_embeddings_batch, rank_by_cosine
import time


def simple_dimension_test():
    """Simple test with different dimension settings."""
    
    test_text = "This is a test message for home equity loans."
    
    print("Simple Dimension Test")
    print("=" * 50)
    
    # Test configurations - adjust these based on available models
    configs = [
        {"model": "amazon.titan-embed-text-v1", "dim": 1536, "name": "Titan v1"},
        {"model": "amazon.titan-embed-text-v1", "dim": 512, "name": "Titan v1 (512 requested)"},  # Will warn and use 1536
    ]
    
    for config in configs:
        try:
            print(f"\nTesting {config['name']}")
            print("-" * 30)
            
            start_time = time.perf_counter()
            embedding = get_titan_embedding(
                test_text, 
                model_id=config["model"], 
                dimensions=config["dim"]
            )
            elapsed = time.perf_counter() - start_time
            
            print(f"✓ Success!")
            print(f"  Requested dimensions: {config['dim']}")
            print(f"  Actual dimensions: {len(embedding)}")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  First 5 values: {embedding[:5]}")
            
        except Exception as e:
            print(f"✗ Error: {e}")


def performance_comparison():
    """Compare performance across different dimension settings."""
    
    # Test data
    messages = [
        "Unlock your home's equity potential today.",
        "Transform your property into financial opportunity.",
        "Access funds through your home's value.",
        "Leverage your home for financial goals.",
        "Turn home equity into available funds."
    ]
    
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    
    # You can test these if you have Titan v2 access:
    test_scenarios = [
        {
            "name": "Titan v1 (1536 fixed)",
            "model_id": "amazon.titan-embed-text-v1",
            "dimensions": 1536,
            "available": True
        },
        # Uncomment these if you have Titan v2:
        # {
        #     "name": "Titan v2 (256 dims)",
        #     "model_id": "amazon.titan-embed-text-v2:0", 
        #     "dimensions": 256,
        #     "available": False  # Set to True if you have access
        # },
        # {
        #     "name": "Titan v2 (512 dims)",
        #     "model_id": "amazon.titan-embed-text-v2:0",
        #     "dimensions": 512,
        #     "available": False  # Set to True if you have access
        # },
        # {
        #     "name": "Titan v2 (1024 dims)",
        #     "model_id": "amazon.titan-embed-text-v2:0",
        #     "dimensions": 1024,
        #     "available": False  # Set to True if you have access
        # }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        if not scenario["available"]:
            print(f"Skipping {scenario['name']} (not available)")
            continue
            
        print(f"\nTesting {scenario['name']}")
        print("-" * 40)
        
        try:
            # Time embedding generation
            start_time = time.perf_counter()
            embeddings = get_titan_embeddings_batch(
                messages, 
                model_id=scenario["model_id"],
                dimensions=scenario["dimensions"]
            )
            embedding_time = time.perf_counter() - start_time
            
            # Time similarity computation
            start_time = time.perf_counter()
            ref_vec = embeddings[0]
            candidates = embeddings[1:]
            ranked = rank_by_cosine(ref_vec, candidates, messages[1:])
            similarity_time = time.perf_counter() - start_time
            
            result = {
                "name": scenario["name"],
                "dimensions": len(embeddings[0]),
                "embedding_time": embedding_time,
                "similarity_time": similarity_time,
                "total_time": embedding_time + similarity_time,
                "top_match": ranked[0]["message"] if ranked else "None"
            }
            results.append(result)
            
            print(f"✓ Dimensions: {result['dimensions']}")
            print(f"  Embedding time: {embedding_time:.3f}s")
            print(f"  Similarity time: {similarity_time:.4f}s")
            print(f"  Total time: {result['total_time']:.3f}s")
            print(f"  Top match: {result['top_match'][:40]}...")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Summary
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print("="*70)
        
        fastest = min(results, key=lambda x: x["total_time"])
        print(f"Fastest configuration: {fastest['name']} ({fastest['total_time']:.3f}s)")
        
        print(f"\nSpeed comparison (relative to fastest):")
        for result in sorted(results, key=lambda x: x["total_time"]):
            relative_speed = result["total_time"] / fastest["total_time"]
            print(f"  {result['name']:20s}: {relative_speed:.2f}x")
        
        print(f"\nDimensions vs Speed:")
        for result in sorted(results, key=lambda x: x["dimensions"]):
            ms_per_dim = result["total_time"] / result["dimensions"] * 1000
            print(f"  {result['dimensions']:4d} dims: {ms_per_dim:.4f} ms/dimension")


def usage_examples():
    """Show common usage patterns."""
    print(f"\n{'='*70}")
    print("USAGE EXAMPLES")
    print("="*70)
    
    print("""
# Basic usage with default dimensions (Titan v1: 1536)
embedding = get_titan_embedding("Your text here")

# Specify dimensions explicitly (will warn if not supported)
embedding = get_titan_embedding("Your text", dimensions=512)

# Use Titan v2 with custom dimensions (if available)
embedding = get_titan_embedding(
    "Your text", 
    model_id="amazon.titan-embed-text-v2:0", 
    dimensions=256
)

# Batch processing with custom dimensions
embeddings = get_titan_embeddings_batch(
    ["Text 1", "Text 2", "Text 3"],
    model_id="amazon.titan-embed-text-v2:0",
    dimensions=512
)

# Trade-offs to consider:
# - Smaller dimensions (256, 512): Faster, less storage, potentially less accurate
# - Larger dimensions (1024, 1536): Slower, more storage, potentially more accurate
# - Titan v1: Only supports 1536 dimensions
# - Titan v2: Supports 256, 512, 1024 dimensions
""")


if __name__ == "__main__":
    simple_dimension_test()
    performance_comparison()
    usage_examples()
