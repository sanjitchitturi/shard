from shard_core import ShardCore


def main():
    shard = ShardCore(
        max_active_tokens=300,
        compression_ratio=0.3,
        alpha=0.4,
        beta=0.3,
        gamma=0.3,
        block_size=4,
    )

    conversation = [
        ("user", "What is machine learning?"),
        ("assistant", "Machine learning is a branch of AI where systems learn patterns from data instead of being explicitly programmed. Common types include supervised, unsupervised, and reinforcement learning."),
        ("user", "Can you explain supervised learning?"),
        ("assistant", "Supervised learning trains a model on labeled examples. The model learns to map inputs to known outputs, then generalizes to new inputs. Classification and regression are the two main supervised tasks."),
        ("user", "Give me an example of classification."),
        ("assistant", "Email spam detection is a classic example. The model is trained on emails labeled spam or not-spam, learns the distinguishing patterns, and can classify new incoming emails."),
        ("user", "What about unsupervised learning?"),
        ("assistant", "Unsupervised learning finds structure in unlabeled data. Clustering algorithms like k-means group similar items together, while dimensionality reduction methods like PCA find compact representations."),
        ("user", "Tell me about neural networks."),
        ("assistant", "Neural networks are layered systems of interconnected nodes inspired by the brain. Each layer transforms its input through learned weights and nonlinear activations. Deep networks stack many such layers."),
        ("user", "What is backpropagation?"),
        ("assistant", "Backpropagation is the algorithm that trains neural networks. It computes how much each weight contributed to the prediction error using the chain rule, then updates weights via gradient descent."),
        ("user", "How does gradient descent work?"),
        ("assistant", "Gradient descent iteratively adjusts parameters in the direction that reduces a loss function. The gradient indicates the steepest ascent, so we move in the opposite direction by a step size called the learning rate."),
        ("user", "What is overfitting?"),
        ("assistant", "Overfitting occurs when a model memorizes training data rather than learning generalizable patterns. It performs well on training but poorly on new data. Regularization, dropout, and more data are common remedies."),
    ]

    print("=" * 70)
    print("SHARD demo: feeding 12 conversational turns")
    print("=" * 70)

    for role, content in conversation:
        shard.add_message(content, role)

    print("\nfinal metrics:")
    metrics = shard.get_metrics()
    print(f"  total messages   : {metrics.total_messages}")
    print(f"  active messages  : {metrics.active_messages}")
    print(f"  memory blocks    : {metrics.memory_blocks}")
    print(f"  raw tokens       : {metrics.raw_tokens}")
    print(f"  current tokens   : {metrics.active_tokens + metrics.compressed_tokens}")
    print(f"  compression ratio: {metrics.compression_ratio}")
    print(f"  tokens saved     : {metrics.tokens_saved}")
    print(f"  avg latency      : {metrics.avg_processing_ms} ms")

    queries = [
        "How do neural networks train?",
        "What is the difference between supervised and unsupervised learning?",
        "How can I prevent overfitting?",
    ]

    print("\nretrieval examples:")
    for q in queries:
        print(f"\n  query: {q}")
        results = shard.retrieve_relevant(q, top_k=3)
        for i, r in enumerate(results, 1):
            preview = r["content"][:80].replace("\n", " ")
            print(f"    {i}. [{r['type']}] score={r['score']:.3f} | {preview}...")

    print("\noptimized model context for 'How does training work?':")
    print("-" * 70)
    print(shard.get_context_for_model(query="How does training work?"))


if __name__ == "__main__":
    main()
