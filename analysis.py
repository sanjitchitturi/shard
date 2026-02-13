# analysis.py
"""
Comprehensive analysis suite for SHARD system
Demonstrates compression efficiency, retrieval accuracy, and scalability
"""

import numpy as np
import matplotlib.pyplot as plt
from shard_core import ShardCore
import time
from typing import List, Dict
import json


class ShardAnalyzer:
    """Analyzes SHARD performance across multiple metrics"""
    
    def __init__(self):
        self.results = {}
    
    def test_compression_efficiency(self, message_counts: List[int] = None):
        """Test compression efficiency at different conversation lengths"""
        if message_counts is None:
            message_counts = [50, 100, 200, 500, 1000]
        
        print("\n" + "=" * 70)
        print("TEST 1: COMPRESSION EFFICIENCY ANALYSIS")
        print("=" * 70)
        
        results = {
            'message_counts': [],
            'compression_ratios': [],
            'tokens_saved': [],
            'memory_blocks': [],
            'effective_multiplier': []
        }
        
        for count in message_counts:
            shard = ShardCore(max_active_tokens=2000, compression_ratio=0.25)
            
            # Generate realistic conversation
            for i in range(count):
                if i % 2 == 0:
                    content = f"User query {i}: Tell me about topic {i % 10} and its implications."
                else:
                    content = f"Assistant response {i}: Here is detailed information about topic {(i-1) % 10} including its background, applications, and significance."
                
                role = 'user' if i % 2 == 0 else 'assistant'
                shard.add_message(content, role)
            
            metrics = shard.get_metrics()
            
            results['message_counts'].append(count)
            results['compression_ratios'].append(float(metrics['compression_ratio'].rstrip('%')))
            results['tokens_saved'].append(metrics['tokens_saved'])
            results['memory_blocks'].append(metrics['memory_blocks'])
            results['effective_multiplier'].append(float(metrics['effective_context_multiplier'].rstrip('x')))
            
            print(f"\nMessages: {count}")
            print(f"  Compression Ratio: {metrics['compression_ratio']}")
            print(f"  Tokens Saved: {metrics['tokens_saved']}")
            print(f"  Memory Blocks: {metrics['memory_blocks']}")
            print(f"  Effective Multiplier: {metrics['effective_context_multiplier']}")
        
        self.results['compression'] = results
        self._plot_compression_results(results)
        return results
    
    def test_retrieval_accuracy(self, num_messages: int = 200):
        """Test semantic retrieval accuracy with known facts"""
        print("\n" + "=" * 70)
        print("TEST 2: RETRIEVAL ACCURACY ANALYSIS")
        print("=" * 70)
        
        shard = ShardCore(max_active_tokens=2000)
        
        # Create conversation with known facts embedded
        facts = [
            ("What is the capital of France?", "The capital of France is Paris."),
            ("Explain quantum computing", "Quantum computing uses quantum mechanics principles like superposition and entanglement for computation."),
            ("What is machine learning?", "Machine learning is a subset of AI that enables systems to learn from data without explicit programming."),
            ("Tell me about photosynthesis", "Photosynthesis is the process by which plants convert light energy into chemical energy."),
            ("What is blockchain?", "Blockchain is a distributed ledger technology that maintains a secure and decentralized record of transactions."),
        ]
        
        # Add messages with filler content between facts
        for i, (question, answer) in enumerate(facts):
            shard.add_message(question, 'user')
            shard.add_message(answer, 'assistant')
            
            # Add filler messages to test retention over distance
            for j in range(15):
                shard.add_message(f"Filler question {i*15+j}", 'user')
                shard.add_message(f"Filler answer {i*15+j}", 'assistant')
        
        # Test retrieval of each fact
        test_queries = [
            ("capital France", "Paris"),
            ("quantum computing", "quantum mechanics"),
            ("machine learning AI", "learn from data"),
            ("photosynthesis plants", "light energy"),
            ("blockchain distributed", "ledger"),
        ]
        
        correct = 0
        results = []
        
        for query, expected_keyword in test_queries:
            retrieved = shard.retrieve_relevant(query, top_k=3)
            
            # Check if expected keyword appears in results
            found = any(expected_keyword.lower() in content.lower() for content, _ in retrieved)
            correct += int(found)
            
            results.append({
                'query': query,
                'expected': expected_keyword,
                'found': found,
                'top_result': retrieved[0][0] if retrieved else "None",
                'score': retrieved[0][1] if retrieved else 0.0
            })
            
            print(f"\nQuery: '{query}'")
            print(f"  Expected keyword: '{expected_keyword}'")
            print(f"  Found: {found}")
            if retrieved:
                print(f"  Top result: {retrieved[0][0][:80]}...")
        
        accuracy = (correct / len(test_queries)) * 100
        print(f"\n{'=' * 70}")
        print(f"RETRIEVAL ACCURACY: {accuracy:.1f}%")
        print(f"{'=' * 70}")
        
        self.results['retrieval'] = {
            'accuracy': accuracy,
            'details': results
        }
        
        return accuracy, results
    
    def test_scalability(self):
        """Test system performance at increasing scales"""
        print("\n" + "=" * 70)
        print("TEST 3: SCALABILITY ANALYSIS")
        print("=" * 70)
        
        message_counts = [100, 500, 1000, 2000, 5000]
        results = {
            'message_counts': [],
            'processing_times': [],
            'memory_usage': [],
            'compression_ratios': []
        }
        
        for count in message_counts:
            shard = ShardCore(max_active_tokens=2000)
            
            start_time = time.time()
            
            for i in range(count):
                content = f"Message {i} with some content about various topics"
                role = 'user' if i % 2 == 0 else 'assistant'
                shard.add_message(content, role)
            
            elapsed = time.time() - start_time
            metrics = shard.get_metrics()
            
            results['message_counts'].append(count)
            results['processing_times'].append(elapsed)
            results['memory_usage'].append(metrics['total_tokens_used'])
            results['compression_ratios'].append(float(metrics['compression_ratio'].rstrip('%')))
            
            print(f"\nMessages: {count}")
            print(f"  Processing Time: {elapsed:.3f}s")
            print(f"  Avg Time per Message: {(elapsed/count)*1000:.2f}ms")
            print(f"  Total Tokens Used: {metrics['total_tokens_used']}")
            print(f"  Compression Ratio: {metrics['compression_ratio']}")
        
        self.results['scalability'] = results
        self._plot_scalability_results(results)
        return results
    
    def test_importance_scoring(self):
        """Analyze importance scoring distribution"""
        print("\n" + "=" * 70)
        print("TEST 4: IMPORTANCE SCORING ANALYSIS")
        print("=" * 70)
        
        shard = ShardCore()
        
        # Messages with varying importance levels
        test_messages = [
            ("Generic greeting", "user"),
            ("I need help with a critical machine learning project deadline tomorrow", "user"),
            ("The quantum entanglement phenomenon was discovered by Einstein", "assistant"),
            ("ok", "user"),
            ("Here is a comprehensive analysis of the neural network architecture including backpropagation algorithms", "assistant"),
            ("thanks", "user"),
        ]
        
        scores = []
        for content, role in test_messages:
            shard.add_message(content, role)
            if shard.active_context:
                scores.append(list(shard.active_context)[-1].importance)
        
        print("\nImportance Scores:")
        for (content, _), score in zip(test_messages, scores):
            print(f"  {score:.3f} - {content[:60]}")
        
        self.results['importance_scores'] = {
            'messages': [c for c, _ in test_messages],
            'scores': scores,
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores))
        }
        
        print(f"\nStatistics:")
        print(f"  Mean: {np.mean(scores):.3f}")
        print(f"  Std Dev: {np.std(scores):.3f}")
        print(f"  Range: [{np.min(scores):.3f}, {np.max(scores):.3f}]")
        
        return scores
    
    def _plot_compression_results(self, results):
        """Generate compression efficiency visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SHARD Compression Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Compression Ratio
        axes[0, 0].plot(results['message_counts'], results['compression_ratios'], 'b-o', linewidth=2)
        axes[0, 0].set_xlabel('Number of Messages')
        axes[0, 0].set_ylabel('Compression Ratio (%)')
        axes[0, 0].set_title('Compression Ratio vs Message Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Tokens Saved
        axes[0, 1].plot(results['message_counts'], results['tokens_saved'], 'g-o', linewidth=2)
        axes[0, 1].set_xlabel('Number of Messages')
        axes[0, 1].set_ylabel('Tokens Saved')
        axes[0, 1].set_title('Total Tokens Saved')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Memory Blocks
        axes[1, 0].plot(results['message_counts'], results['memory_blocks'], 'r-o', linewidth=2)
        axes[1, 0].set_xlabel('Number of Messages')
        axes[1, 0].set_ylabel('Memory Blocks Created')
        axes[1, 0].set_title('Memory Blocks vs Message Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Effective Multiplier
        axes[1, 1].plot(results['message_counts'], results['effective_multiplier'], 'm-o', linewidth=2)
        axes[1, 1].set_xlabel('Number of Messages')
        axes[1, 1].set_ylabel('Context Multiplier (x)')
        axes[1, 1].set_title('Effective Context Extension')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('shard_compression_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\nCompression analysis plot saved as 'shard_compression_analysis.png'")
    
    def _plot_scalability_results(self, results):
        """Generate scalability visualizations"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('SHARD Scalability Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Processing Time
        axes[0].plot(results['message_counts'], results['processing_times'], 'b-o', linewidth=2)
        axes[0].set_xlabel('Number of Messages')
        axes[0].set_ylabel('Processing Time (seconds)')
        axes[0].set_title('Processing Time vs Message Count')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Memory Usage
        axes[1].plot(results['message_counts'], results['memory_usage'], 'g-o', linewidth=2)
        axes[1].set_xlabel('Number of Messages')
        axes[1].set_ylabel('Total Tokens Used')
        axes[1].set_title('Memory Usage vs Message Count')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('shard_scalability_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Scalability analysis plot saved as 'shard_scalability_analysis.png'")
    
    def save_results(self, filename='shard_analysis_results.json'):
        """Save all analysis results to JSON file"""
        def convert(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        serializable_results = json.loads(
            json.dumps(self.results, default=convert)
        )
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to '{filename}'")
    
    def run_full_analysis(self):
        """Execute all analysis tests"""
        print("\n" + "=" * 70)
        print("SHARD COMPREHENSIVE ANALYSIS SUITE")
        print("=" * 70)
        
        self.test_compression_efficiency()
        self.test_retrieval_accuracy()
        self.test_scalability()
        self.test_importance_scoring()
        
        self.save_results()
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - shard_compression_analysis.png")
        print("  - shard_scalability_analysis.png")
        print("  - shard_analysis_results.json")


if __name__ == "__main__":
    analyzer = ShardAnalyzer()
    analyzer.run_full_analysis()