# shard_simple.py
"""
Simplified SHARD implementation for learning core concepts
Focuses on the main idea: keep recent messages active, compress old ones
"""

import numpy as np
from collections import deque


class SimplifiedShard:
    """
    Simplified version of SHARD for easy understanding.
    Core concept: maintain a sliding window of recent messages,
    compress older messages into summaries.
    """
    
    def __init__(self, max_active_messages=20):
        """
        Initialize with a maximum number of active messages
        
        Args:
            max_active_messages: How many recent messages to keep active
        """
        self.max_active_messages = max_active_messages
        self.active_messages = deque(maxlen=max_active_messages)
        self.compressed_summaries = []
        
    def add_message(self, text, role='user'):
        """
        Add a message and trigger compression when limit reached
        
        Args:
            text: Message content
            role: 'user' or 'assistant'
        """
        message = {
            'text': text,
            'role': role,
            'id': len(self.active_messages)
        }
        self.active_messages.append(message)
        
        # Compress when we have too many active messages
        if len(self.active_messages) >= self.max_active_messages:
            self._compress_old_messages()
    
    def _compress_old_messages(self):
        """
        Create a simple summary of the oldest messages.
        In this simplified version, we just concatenate them.
        """
        # Take the oldest 5 messages
        to_compress = list(self.active_messages)[:5]
        
        # Create simple extractive summary
        summary = " | ".join([f"{m['role']}: {m['text'][:50]}" for m in to_compress])
        self.compressed_summaries.append(summary)
        
        print(f"Compressed {len(to_compress)} messages into summary")
    
    def get_full_context(self):
        """
        Get the complete context including both summaries and active messages.
        This is what would be passed to the language model.
        """
        context = []
        
        # Add compressed past context
        if self.compressed_summaries:
            context.append("PAST CONTEXT (Compressed)")
            for summary in self.compressed_summaries[-3:]:  # Last 3 summaries only
                context.append(summary)
        
        # Add recent active messages
        context.append("\nRECENT MESSAGES (Active)")
        for msg in self.active_messages:
            context.append(f"[{msg['role']}]: {msg['text']}")
        
        return "\n".join(context)
    
    def search(self, query):
        """
        Simple keyword-based search in active messages.
        Returns messages that share words with the query.
        """
        results = []
        query_words = set(query.lower().split())
        
        # Check each active message for word overlap
        for msg in self.active_messages:
            msg_words = set(msg['text'].lower().split())
            overlap = len(query_words & msg_words)
            if overlap > 0:
                results.append((msg['text'], overlap))
        
        # Sort by number of matching words and return top 3
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:3]
    
    def get_stats(self):
        """Get basic statistics about the system state"""
        return {
            'active_messages': len(self.active_messages),
            'compressed_summaries': len(self.compressed_summaries),
            'total_handled': len(self.active_messages) + len(self.compressed_summaries) * 5
        }


def demo_simplified():
    """Demonstration of the simplified SHARD system"""
    print("=" * 70)
    print("SIMPLIFIED SHARD DEMO")
    print("Understanding the Core Concept")
    print("=" * 70)
    
    shard = SimplifiedShard(max_active_messages=10)
    
    # Sample conversation
    conversation = [
        "What is Python?",
        "Python is a programming language created by Guido van Rossum.",
        "What about JavaScript?",
        "JavaScript is used for web development and runs in browsers.",
        "Tell me about machine learning",
        "Machine learning enables computers to learn from data.",
        "What is deep learning?",
        "Deep learning uses neural networks with many layers.",
        "Explain neural networks",
        "Neural networks are inspired by the human brain structure.",
        "What about AI?",
        "AI is the simulation of human intelligence in machines.",
        "Tell me about data science",
        "Data science combines statistics, programming, and domain knowledge.",
        "What is Python used for?",
        "Python is used for web dev, data science, AI, and automation.",
    ]
    
    print("\n1. Adding messages to conversation:")
    print("-" * 70)
    for i, text in enumerate(conversation):
        role = 'user' if i % 2 == 0 else 'assistant'
        shard.add_message(text, role)
        print(f"[{i+1:2d}] {role:10s}: {text[:50]}")
    
    print("\n2. System Statistics:")
    print("-" * 70)
    stats = shard.get_stats()
    for key, value in stats.items():
        print(f"  {key:25s}: {value}")
    
    print("\n3. Search Test:")
    print("-" * 70)
    query = "Python programming"
    print(f"  Query: '{query}'")
    results = shard.search(query)
    for i, (text, score) in enumerate(results, 1):
        print(f"  {i}. [Score: {score}] {text}")
    
    print("\n4. Full context for model:")
    print("-" * 70)
    print(shard.get_full_context())
    
    print("\n" + "=" * 70)
    print("KEY CONCEPT: Keep recent messages active, compress old ones")
    print("This allows handling much longer conversations efficiently")
    print("=" * 70)


if __name__ == "__main__":
    demo_simplified()