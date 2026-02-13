# comparison_demo.py
"""
Side-by-side comparison of Full SHARD vs Simplified SHARD
Demonstrates the differences between implementations
"""

from shard_core import ShardCore
from shard_simple import SimplifiedShard
import time


def run_comparison_demo():
    """Run side-by-side comparison of both implementations"""
    print("=" * 80)
    print("SHARD COMPARISON: Full vs Simplified Implementation")
    print("=" * 80)
    
    # Initialize both systems
    shard_full = ShardCore(max_active_tokens=1000, compression_ratio=0.3)
    shard_simple = SimplifiedShard(max_active_messages=15)
    
    # Test conversation
    conversation = [
        ("What is machine learning?", "user"),
        ("Machine learning is a subset of AI that enables systems to learn from data.", "assistant"),
        ("Tell me about neural networks", "user"),
        ("Neural networks are computing systems inspired by biological neural networks.", "assistant"),
        ("What about deep learning?", "user"),
        ("Deep learning uses multi-layer neural networks for complex pattern recognition.", "assistant"),
        ("Explain backpropagation", "user"),
        ("Backpropagation is an algorithm for training neural networks using gradient descent.", "assistant"),
        ("What is supervised learning?", "user"),
        ("Supervised learning uses labeled training data to teach models.", "assistant"),
        ("And unsupervised learning?", "user"),
        ("Unsupervised learning finds patterns in unlabeled data.", "assistant"),
        ("What about reinforcement learning?", "user"),
        ("Reinforcement learning trains agents through rewards and penalties.", "assistant"),
        ("Tell me about transformers", "user"),
        ("Transformers use self-attention mechanisms for sequence processing.", "assistant"),
        ("What is GPT?", "user"),
        ("GPT is a generative pretrained transformer model for text generation.", "assistant"),
        ("Explain attention mechanisms", "user"),
        ("Attention allows models to focus on relevant parts of input sequences.", "assistant"),
    ]
    
    print("\n1. ADDING MESSAGES")
    print("-" * 80)
    
    # Time both approaches
    start = time.time()
    for content, role in conversation:
        shard_full.add_message(content, role)
    full_time = time.time() - start
    
    start = time.time()
    for content, role in conversation:
        shard_simple.add_message(content, role)
    simple_time = time.time() - start
    
    print(f"Full SHARD processing time: {full_time*1000:.2f}ms")
    print(f"Simplified SHARD processing time: {simple_time*1000:.2f}ms")
    
    print("\n2. METRICS COMPARISON")
    print("-" * 80)
    
    full_metrics = shard_full.get_metrics()
    simple_stats = shard_simple.get_stats()
    
    print("\nFull SHARD:")
    print(f"  Active Context Size: {full_metrics['active_context_size']}")
    print(f"  Memory Blocks: {full_metrics['memory_blocks']}")
    print(f"  Compression Ratio: {full_metrics['compression_ratio']}")
    print(f"  Tokens Saved: {full_metrics['tokens_saved']}")
    print(f"  Features: Semantic embeddings, importance scoring, temporal decay")
    
    print("\nSimplified SHARD:")
    print(f"  Active Messages: {simple_stats['active_messages']}")
    print(f"  Compressed Summaries: {simple_stats['compressed_summaries']}")
    print(f"  Total Handled: {simple_stats['total_handled']}")
    print(f"  Features: Basic compression only")
    
    print("\n3. RETRIEVAL TEST")
    print("-" * 80)
    
    query = "neural networks machine learning"
    
    print(f"\nQuery: '{query}'")
    print("\nFull SHARD Results:")
    start = time.time()
    full_results = shard_full.retrieve_relevant(query, top_k=3)
    full_retrieval_time = time.time() - start
    
    for i, (content, score) in enumerate(full_results, 1):
        print(f"  {i}. [Score: {score:.3f}] {content[:60]}...")
    print(f"  Retrieval time: {full_retrieval_time*1000:.2f}ms")
    
    print("\nSimplified SHARD Results:")
    start = time.time()
    simple_results = shard_simple.search(query)
    simple_retrieval_time = time.time() - start
    
    for i, (content, score) in enumerate(simple_results, 1):
        print(f"  {i}. [Score: {score}] {content[:60]}...")
    print(f"  Retrieval time: {simple_retrieval_time*1000:.2f}ms")
    
    print("\n4. CONTEXT GENERATION")
    print("-" * 80)
    
    print("\nFull SHARD Context:")
    print("-" * 40)
    full_context = shard_full.get_context_for_model()
    print(full_context[:500] + "..." if len(full_context) > 500 else full_context)
    
    print("\n\nSimplified SHARD Context:")
    print("-" * 40)
    simple_context = shard_simple.get_full_context()
    print(simple_context[:500] + "..." if len(simple_context) > 500 else simple_context)
    
    print("\n5. KEY DIFFERENCES")
    print("=" * 80)
    print("""
Full SHARD:
  + Semantic embeddings for accurate retrieval
  + Multi-factor importance scoring (relevance, novelty, temporal)
  + Mathematical optimization
  + High accuracy retrieval
  + Configurable compression strategies
  - More complex implementation
    
Simplified SHARD:
  + Easy to understand and implement
  + Fast processing
  + Low overhead
  + Good for learning core concepts
  - Less accurate retrieval
  - Basic compression only
    
Recommendation:
  - Use Full SHARD for production systems requiring high accuracy
  - Use Simplified SHARD for learning or prototypes
    """)


if __name__ == "__main__":
    run_comparison_demo()