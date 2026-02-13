# test_scenarios.py
"""
Unit and integration tests for SHARD system
Validates correctness and performance
"""

from shard_core import ShardCore
import unittest


class TestShardCore(unittest.TestCase):
    """Unit tests for SHARD core functionality"""
    
    def setUp(self):
        """Initialize test instance"""
        self.shard = ShardCore(max_active_tokens=1000)
    
    def test_message_addition(self):
        """Test basic message addition"""
        self.shard.add_message("Test message", "user")
        self.assertEqual(len(self.shard.active_context), 1)
        self.assertEqual(self.shard.metrics['total_messages'], 1)
    
    def test_compression_triggers(self):
        """Test that compression triggers at token limit"""
        for i in range(50):
            self.shard.add_message(f"Message {i} with some content", "user")
        
        self.assertGreater(len(self.shard.memory_blocks), 0)
        self.assertGreater(self.shard.metrics['compressed_blocks'], 0)
    
    def test_importance_scoring(self):
        """Test importance scoring consistency"""
        self.shard.add_message("Critical urgent deadline information", "user")
        high_importance_msg = list(self.shard.active_context)[-1]
        
        self.shard.add_message("ok", "user")
        low_importance_msg = list(self.shard.active_context)[-1]
        
        self.assertGreater(high_importance_msg.importance, low_importance_msg.importance)
    
    def test_retrieval(self):
        """Test semantic retrieval functionality"""
        self.shard.add_message("Python is a programming language", "assistant")
        self.shard.add_message("The capital of France is Paris", "assistant")
        
        for i in range(20):
            self.shard.add_message(f"Random content {i}", "user")
        
        results = self.shard.retrieve_relevant("programming language", top_k=2)
        
        self.assertTrue(any("Python" in content for content, _ in results))
    
    def test_token_estimation(self):
        """Test token estimation accuracy"""
        text = "This is a test message"
        tokens = self.shard._estimate_tokens(text)
        self.assertGreater(tokens, 0)
        self.assertAlmostEqual(tokens, len(text.split()) * 1.3, delta=2)
    
    def test_embedding_generation(self):
        """Test embedding generation"""
        embedding = self.shard._create_embedding("test message")
        self.assertEqual(embedding.shape, (384,))
        self.assertAlmostEqual(sum(embedding ** 2) ** 0.5, 1.0, places=5)


class TestShardScenarios(unittest.TestCase):
    """Integration tests for real-world scenarios"""
    
    def test_long_conversation(self):
        """Test handling a long conversation"""
        shard = ShardCore(max_active_tokens=1000)
        
        for i in range(200):
            shard.add_message(f"Message {i}", "user" if i % 2 == 0 else "assistant")
        
        metrics = shard.get_metrics()
        
        self.assertGreater(metrics['memory_blocks'], 0)
        self.assertGreater(metrics['tokens_saved'], 0)
        self.assertLess(metrics['active_context_size'], 200)
    
    def test_context_retrieval_accuracy(self):
        """Test that important information is retrievable"""
        shard = ShardCore(max_active_tokens=800)
        
        important_fact = "The chemical formula for water is H2O"
        shard.add_message(important_fact, "assistant")
        
        for i in range(100):
            shard.add_message(f"Irrelevant message {i}", "user")
        
        results = shard.retrieve_relevant("chemical formula water", top_k=3)
        
        self.assertTrue(any("H2O" in content for content, _ in results))
    
    def test_compression_ratio(self):
        """Test compression achieves target ratio"""
        shard = ShardCore(max_active_tokens=1000, compression_ratio=0.3)
        
        for i in range(100):
            shard.add_message(f"Test message {i} with some content about topic {i}", "user")
        
        metrics = shard.get_metrics()
        
        compression_pct = float(metrics['compression_ratio'].rstrip('%'))
        self.assertLess(compression_pct, 50)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()