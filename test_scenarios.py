import math
import sys
import random
from shard_core import ShardCore, _hash_embedding, _cosine, _shannon_entropy


def assert_eq(actual, expected, label):
    status = "PASS" if actual == expected else "FAIL"
    print(f"  [{status}] {label}: expected={expected}, actual={actual}")
    return actual == expected


def assert_near(actual, expected, tol, label):
    ok = abs(actual - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: expected~{expected}, actual={actual:.4f}, tol={tol}")
    return ok


def assert_true(cond, label):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def test_invalid_weights():
    print("\ntest_invalid_weights")
    try:
        ShardCore(alpha=0.5, beta=0.5, gamma=0.5)
        return assert_true(False, "should have raised ValueError")
    except ValueError:
        return assert_true(True, "correctly rejected weights that don't sum to 1.0")


def test_basic_add():
    print("\ntest_basic_add")
    s = ShardCore(max_active_tokens=10000)
    s.add_message("hello world", "user")
    s.add_message("hi there", "assistant")
    ok1 = assert_eq(len(s.active), 2, "two messages added to active")
    ok2 = assert_eq(len(s.memory), 0, "no compression yet")
    ok3 = assert_eq(s.get_metrics().total_messages, 2, "metrics report 2 messages")
    return ok1 and ok2 and ok3


def test_compression_triggers():
    print("\ntest_compression_triggers")
    s = ShardCore(max_active_tokens=50, block_size=3)
    for i in range(20):
        s.add_message(f"this is message number {i} with some additional padding text " * 3, "user")
    metrics = s.get_metrics()
    ok1 = assert_true(metrics.memory_blocks > 0, "compression triggered at least once")
    ok2 = assert_true(metrics.tokens_saved > 0, "tokens were saved by compression")
    ok3 = assert_true(metrics.compression_ratio < 1.0, "compression ratio below 1.0")
    return ok1 and ok2 and ok3


def test_retrieval_relevance():
    print("\ntest_retrieval_relevance")
    s = ShardCore(max_active_tokens=10000)
    s.add_message("python is a great programming language for beginners", "user")
    s.add_message("the weather is sunny today and quite warm outside", "user")
    s.add_message("javascript runs in browsers and supports asynchronous code", "user")
    s.add_message("it might rain heavily tomorrow afternoon", "user")

    results = s.retrieve_relevant("python programming language tutorial", top_k=2)
    top_content = results[0]["content"].lower()
    ok = assert_true(
        "python" in top_content or "programming" in top_content,
        "top result is programming related when query shares vocabulary",
    )
    return ok


def test_embedding_determinism():
    print("\ntest_embedding_determinism")
    e1 = _hash_embedding("hello world")
    e2 = _hash_embedding("hello world")
    e3 = _hash_embedding("different text")
    ok1 = assert_near(_cosine(e1, e2), 1.0, 1e-6, "same text -> identical embedding")
    ok2 = assert_true(_cosine(e1, e3) < 0.99, "different text -> different embedding")
    return ok1 and ok2


def test_entropy():
    print("\ntest_entropy")
    e1 = _shannon_entropy("a a a a a a")
    e2 = _shannon_entropy("the quick brown fox jumps over lazy dog")
    ok1 = assert_near(e1, 0.0, 1e-6, "uniform text has zero entropy")
    ok2 = assert_true(e2 > 2.0, "diverse text has high entropy")
    return ok1 and ok2


def test_scale_1000_messages():
    print("\ntest_scale_1000_messages")
    s = ShardCore(max_active_tokens=1000, block_size=8)
    random.seed(42)
    topics = ["python", "music", "cooking", "history", "science", "travel"]
    for i in range(1000):
        topic = random.choice(topics)
        s.add_message(f"discussing {topic} topic with example {i} and context details", "user")
    m = s.get_metrics()
    ok1 = assert_eq(m.total_messages, 1000, "all 1000 messages processed")
    ok2 = assert_true(m.memory_blocks > 50, "many memory blocks formed")
    ok3 = assert_true(math.isfinite(m.avg_processing_ms), "processing time is finite")
    ok4 = assert_true(m.compression_ratio < 0.5, "achieved >50% compression")
    return ok1 and ok2 and ok3 and ok4


def test_reset():
    print("\ntest_reset")
    s = ShardCore(max_active_tokens=50, block_size=3)
    for i in range(10):
        s.add_message(f"message {i} with padding text content " * 3, "user")
    s.reset()
    m = s.get_metrics()
    ok1 = assert_eq(m.total_messages, 0, "reset clears message count")
    ok2 = assert_eq(m.memory_blocks, 0, "reset clears memory blocks")
    ok3 = assert_eq(m.active_messages, 0, "reset clears active")
    return ok1 and ok2 and ok3


def main():
    tests = [
        test_invalid_weights,
        test_basic_add,
        test_compression_triggers,
        test_retrieval_relevance,
        test_embedding_determinism,
        test_entropy,
        test_scale_1000_messages,
        test_reset,
    ]
    passed = 0
    for t in tests:
        if t():
            passed += 1
    print(f"\n{'=' * 50}")
    print(f"results: {passed}/{len(tests)} tests passed")
    print("=" * 50)
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
