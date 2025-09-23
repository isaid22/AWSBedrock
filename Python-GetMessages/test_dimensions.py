#!/usr/bin/env python3
"""
Utility script to test different Titan embedding dimensions and compare their performance.
"""

import time
import json
from pathlib import Path
from embedding_work import (
    get_titan_embedding, 
    get_titan_embeddings_batch, 
    rank_by_cosine,
    build_concatenated_prompt_from_yaml
)


def test_embedding_dimensions():
    """Test different Titan embedding dimensions and compare results."""
    
    # Load your prompt and candidate messages
    config_file = Path("run_config.yaml")
    prompt = build_concatenated_prompt_from_yaml(config_file)
    
    candidate_messages = [
        "Unlock your home's potential with a flexible loan.",
        "Transform your home into a financial opportunity.",
        "Access your home equity for your dreams.",
        "Empower your financial future with your home.",
        "Turn your home into a resource for your goals.",
    ]
    
    # Define test configurations
    test_configs = [
        {
            "model_id": "amazon.titan-embed-text-v1", 
            "dimensions": 1536, 
            "name": "Titan v1",
            "description": "Original Titan model with fixed 1536 dimensions"
        },
        # Uncomment these if you have access to Titan v2
        # {
        #     "model_id": "amazon.titan-embed-text-v2:0", 
        #     "dimensions": 256, 
        #     "name": "Titan v2 (256)",
        #     "description": "Titan v2 with 256 dimensions - faster, less detailed"
        # },
        # {
        #     "model_id": "amazon.titan-embed-text-v2:0", 
        #     "dimensions": 512, 
        #     "name": "Titan v2 (512)",
        #     "description": "Titan v2 with 512 dimensions - balanced speed/quality"
        # },
        # {
        #     "model_id": "amazon.titan-embed-text-v2:0", 
        #     "dimensions": 1024, 
        #     "name": "Titan v2 (1024)",
        #     "description": "Titan v2 with 1024 dimensions - high quality"
        # },
    ]
    
    print("Testing Different Titan Embedding Dimensions")
    print("=" * 80)
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Number of candidates: {len(candidate_messages)}")
    print()
    
    results = []
    
    for config in test_configs:
        print(f"Testing {config['name']}")
        print(f"Description: {config['description']}")
        print("-" * 60)
        
        try:
            # Time the embedding generation
            start_time = time.perf_counter()
            
            # Get embeddings for prompt and candidates
            texts = [prompt] + candidate_messages
            vecs = get_titan_embeddings_batch(
                texts, 
                model_id=config["model_id"], 
                dimensions=config["dimensions"]
            )
            
            embedding_time = time.perf_counter() - start_time
            
            # Time the similarity ranking
            start_time = time.perf_counter()
            ref_vec = vecs[0]
            cand_vecs = vecs[1:]
            ranked = rank_by_cosine(ref_vec, cand_vecs, candidate_messages)
            ranking_time = time.perf_counter() - start_time
            
            # Store results
            result = {
                "config": config,
                "embedding_time": embedding_time,
                "ranking_time": ranking_time,
                "total_time": embedding_time + ranking_time,
                "actual_dimensions": len(vecs[0]) if vecs and vecs[0] else 0,
                "top_3_similarities": [
                    {
                        "message": item["message"][:50] + "...",
                        "similarity": round(item["cosine_similarity"], 4)
                    }
                    for item in ranked[:3]
                ]
            }
            results.append(result)
            
            # Print results
            print(f"✓ Success!")
            print(f"  Actual dimensions: {result['actual_dimensions']}")
            print(f"  Embedding time: {embedding_time:.3f}s")
            print(f"  Ranking time: {ranking_time:.3f}s")
            print(f"  Total time: {result['total_time']:.3f}s")
            print(f"  Top candidate: {ranked[0]['message'][:50]}... (sim: {ranked[0]['cosine_similarity']:.4f})")
            print()
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            print()
            continue
    
    # Print comparison summary
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        
        # Sort by total time
        results_by_speed = sorted(results, key=lambda x: x["total_time"])
        fastest = results_by_speed[0]
        
        print("Performance Ranking (fastest to slowest):")
        for i, result in enumerate(results_by_speed):
            config = result["config"]
            speedup = fastest["total_time"] / result["total_time"]
            print(f"  {i+1}. {config['name']}: {result['total_time']:.3f}s ({speedup:.2f}x vs fastest)")
        
        print(f"\nDimensions vs Performance:")
        for result in sorted(results, key=lambda x: x["actual_dimensions"]):
            config = result["config"]
            dims = result["actual_dimensions"]
            time_per_dim = result["total_time"] / dims * 1000  # ms per dimension
            print(f"  {dims:4d} dims: {result['total_time']:.3f}s ({time_per_dim:.3f} ms/dim)")
        
        print(f"\nSimilarity Ranking Consistency:")
        print("(Check if different dimensions give similar ranking results)")
        if len(results) >= 2:
            base_result = results[0]
            base_top3 = [item["message"] for item in base_result["top_3_similarities"]]
            
            for result in results[1:]:
                config = result["config"]
                top3 = [item["message"] for item in result["top_3_similarities"]]
                matches = sum(1 for msg in top3 if msg in base_top3)
                consistency = matches / 3 * 100
                print(f"  {config['name']} vs {base_result['config']['name']}: {consistency:.0f}% top-3 match")


def create_dimension_test_config():
    """Create a configuration file for easy dimension testing."""
    config = {
        "embedding_configs": [
            {
                "name": "titan_v1_default",
                "model_id": "amazon.titan-embed-text-v1",
                "dimensions": 1536,
                "description": "Standard Titan v1 with 1536 dimensions"
            },
            {
                "name": "titan_v2_small",
                "model_id": "amazon.titan-embed-text-v2:0",
                "dimensions": 256,
                "description": "Titan v2 with reduced dimensions for speed"
            },
            {
                "name": "titan_v2_medium",
                "model_id": "amazon.titan-embed-text-v2:0",
                "dimensions": 512,
                "description": "Titan v2 with balanced dimensions"
            },
            {
                "name": "titan_v2_large",
                "model_id": "amazon.titan-embed-text-v2:0",
                "dimensions": 1024,
                "description": "Titan v2 with high-quality dimensions"
            }
        ],
        "usage_notes": [
            "Smaller dimensions (256, 512) are faster but may be less accurate",
            "Larger dimensions (1024, 1536) are slower but more accurate",
            "Titan v1 only supports 1536 dimensions",
            "Titan v2 supports 256, 512, and 1024 dimensions"
        ]
    }
    
    config_path = Path("embedding_dimension_configs.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Created configuration file: {config_path}")
    print("Edit this file to customize your test configurations!")


if __name__ == "__main__":
    print("Titan Embedding Dimension Tester")
    print("=" * 50)
    print()
    
    # Create config file for future use
    create_dimension_test_config()
    print()
    
    # Run the tests
    test_embedding_dimensions()
