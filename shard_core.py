# shard_core.py
"""
SHARD: Semantic Hierarchical Archive with Retrieval and Distillation
Core system for enabling longer context in LLM chats

Mathematical Foundation:
1. Importance Scoring: I(m) = a*R(m) + b*N(m) + c*T(m)
   R(m) = Relevance score (semantic similarity to current context)
   N(m) = Novelty score (information entropy)
   T(m) = Temporal decay (forgetting curve)

2. Memory Compression: C(M) = sum of weighted summaries
   Hierarchical summarization with importance-based weighting

3. Retrieval: Top-k items by similarity S(q, m) = cosine(E(q), E(m))
   E() is the embedding function
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import hashlib
import math


@dataclass
class Message:
    """Represents a single message in the conversation"""
    content: str
    timestamp: int
    role: str
    importance: float = 0.0
    embedding: Optional[np.ndarray] = None
    id: str = field(default_factory=lambda: hashlib.md5(str(np.random.random()).encode()).hexdigest()[:8])


@dataclass
class MemoryBlock:
    """Compressed memory block containing multiple messages"""
    summary: str
    messages: List[Message]
    importance: float
    timestamp_range: Tuple[int, int]
    embedding: Optional[np.ndarray] = None


class ShardCore:
    """Core SHARD system for long context management"""
    
    def __init__(
        self,
        max_active_tokens: int = 4000,
        max_memory_blocks: int = 50,
        compression_ratio: float = 0.3,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3
    ):
        """
        Initialize SHARD system
        
        Args:
            max_active_tokens: Maximum tokens to keep in active context
            max_memory_blocks: Maximum number of compressed memory blocks
            compression_ratio: Target compression ratio (0.3 = 30% of original)
            alpha: Weight for relevance in importance scoring
            beta: Weight for novelty in importance scoring
            gamma: Weight for temporal decay in importance scoring
        """
        self.max_active_tokens = max_active_tokens
        self.max_memory_blocks = max_memory_blocks
        self.compression_ratio = compression_ratio
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Storage structures
        self.active_context = deque(maxlen=100)
        self.memory_blocks = []
        self.semantic_index = {}
        
        # Performance tracking
        self.metrics = {
            'total_messages': 0,
            'compressed_blocks': 0,
            'tokens_saved': 0,
            'retrieval_accuracy': []
        }
        
        self.current_time = 0
        self._embedding_cache = {}
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding vector using hash-based projection.
        In production, replace with sentence-transformers or OpenAI embeddings.
        """
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        words = text.lower().split()
        embedding = np.zeros(384, dtype=np.float32)
        
        # Hash each word to multiple dimensions for better distribution
        for word in words:
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx1 = hash_val % 384
            idx2 = (hash_val // 384) % 384
            idx3 = (hash_val // (384 * 384)) % 384
            embedding[[idx1, idx2, idx3]] += 1.0
        
        # L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        self._embedding_cache[text] = embedding
        return embedding
    
    def _calculate_relevance(self, msg: Message, context_embedding: np.ndarray) -> float:
        """Calculate semantic relevance to current context using cosine similarity"""
        if msg.embedding is None:
            msg.embedding = self._create_embedding(msg.content)
        
        similarity = np.dot(msg.embedding, context_embedding)
        return max(0.0, float(similarity))
    
    def _calculate_novelty(self, msg: Message) -> float:
        """
        Calculate information novelty using entropy-based metric.
        Higher entropy means more diverse/novel content.
        Formula: H(m) = -sum(p(w) * log2(p(w)))
        """
        words = msg.content.lower().split()
        if not words:
            return 0.0
        
        # Calculate word frequency distribution
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Calculate Shannon entropy
        total = len(words)
        entropy = 0.0
        for freq in word_freq.values():
            p = freq / total
            entropy -= p * math.log2(p + 1e-10)
        
        # Normalize by maximum possible entropy
        max_entropy = math.log2(total) if total > 1 else 1.0
        return entropy / max_entropy
    
    def _calculate_temporal_decay(self, msg: Message) -> float:
        """
        Apply Ebbinghaus forgetting curve: R(t) = e^(-t/S)
        where t is time elapsed and S is the memory strength constant
        """
        time_elapsed = self.current_time - msg.timestamp
        strength = 5.0
        return math.exp(-time_elapsed / strength)
    
    def calculate_importance(self, msg: Message, context_embedding: np.ndarray) -> float:
        """
        Calculate overall importance score using weighted combination
        I(m) = alpha*R(m) + beta*N(m) + gamma*T(m)
        """
        relevance = self._calculate_relevance(msg, context_embedding)
        novelty = self._calculate_novelty(msg)
        temporal = self._calculate_temporal_decay(msg)
        
        importance = (
            self.alpha * relevance +
            self.beta * novelty +
            self.gamma * temporal
        )
        
        msg.importance = importance
        return importance
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using rough approximation"""
        return int(len(text.split()) * 1.3)
    
    def _create_summary(self, messages: List[Message]) -> str:
        """
        Create extractive summary based on importance scores.
        Selects most important messages until target compression ratio reached.
        """
        if not messages:
            return ""
        
        sorted_msgs = sorted(messages, key=lambda m: m.importance, reverse=True)
        
        total_tokens = sum(self._estimate_tokens(m.content) for m in messages)
        target_tokens = int(total_tokens * self.compression_ratio)
        
        summary_parts = []
        current_tokens = 0
        
        for msg in sorted_msgs:
            msg_tokens = self._estimate_tokens(msg.content)
            if current_tokens + msg_tokens <= target_tokens:
                summary_parts.append(f"[{msg.role}]: {msg.content}")
                current_tokens += msg_tokens
            else:
                break
        
        return " | ".join(summary_parts) if summary_parts else messages[0].content[:100]
    
    def add_message(self, content: str, role: str = 'user'):
        """Add a new message to the context and compress if needed"""
        self.current_time += 1
        
        msg = Message(
            content=content,
            timestamp=self.current_time,
            role=role
        )
        msg.embedding = self._create_embedding(content)
        
        # Calculate context embedding from recent messages
        if self.active_context:
            recent_embeddings = [m.embedding for m in list(self.active_context)[-5:] 
                               if m.embedding is not None]
            context_embedding = np.mean(recent_embeddings, axis=0) if recent_embeddings else msg.embedding
        else:
            context_embedding = msg.embedding
        
        # Calculate and assign importance score
        self.calculate_importance(msg, context_embedding)
        
        self.active_context.append(msg)
        self.metrics['total_messages'] += 1
        
        # Trigger compression if token limit exceeded
        current_tokens = sum(self._estimate_tokens(m.content) for m in self.active_context)
        if current_tokens > self.max_active_tokens:
            self._compress_context()
    
    def _compress_context(self):
        """Compress older messages into memory blocks"""
        compress_count = len(self.active_context) // 2
        if compress_count < 5:
            return
        
        # Extract messages to compress
        to_compress = [self.active_context.popleft() for _ in range(compress_count)]
        
        # Create compressed memory block
        summary = self._create_summary(to_compress)
        block = MemoryBlock(
            summary=summary,
            messages=to_compress,
            importance=np.mean([m.importance for m in to_compress]),
            timestamp_range=(to_compress[0].timestamp, to_compress[-1].timestamp),
            embedding=self._create_embedding(summary)
        )
        
        self.memory_blocks.append(block)
        self.metrics['compressed_blocks'] += 1
        
        # Track compression savings
        original_tokens = sum(self._estimate_tokens(m.content) for m in to_compress)
        compressed_tokens = self._estimate_tokens(summary)
        self.metrics['tokens_saved'] += (original_tokens - compressed_tokens)
        
        # Prune old blocks if limit exceeded
        if len(self.memory_blocks) > self.max_memory_blocks:
            self.memory_blocks.sort(key=lambda b: b.importance, reverse=True)
            self.memory_blocks = self.memory_blocks[:self.max_memory_blocks]
    
    def retrieve_relevant(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve most relevant information for a query using semantic search.
        Returns list of (content, relevance_score) tuples.
        """
        query_embedding = self._create_embedding(query)
        results = []
        
        # Search active context
        for msg in self.active_context:
            if msg.embedding is not None:
                score = float(np.dot(query_embedding, msg.embedding))
                results.append((msg.content, score))
        
        # Search memory blocks
        for block in self.memory_blocks:
            if block.embedding is not None:
                score = float(np.dot(query_embedding, block.embedding))
                results.append((block.summary, score))
        
        # Sort by relevance and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_context_for_model(self) -> str:
        """
        Build optimized context string for model consumption.
        Combines relevant memory blocks with active conversation.
        """
        context_parts = []
        
        # Include most important memory blocks
        if self.memory_blocks:
            top_blocks = sorted(self.memory_blocks, key=lambda b: b.importance, reverse=True)[:3]
            context_parts.append("RELEVANT MEMORY")
            for block in top_blocks:
                context_parts.append(
                    f"[Past conversation, timestamps {block.timestamp_range[0]}-{block.timestamp_range[1]}]: "
                    f"{block.summary}"
                )
        
        # Include active context
        context_parts.append("\nACTIVE CONVERSATION")
        for msg in self.active_context:
            context_parts.append(f"[{msg.role}]: {msg.content}")
        
        return "\n".join(context_parts)
    
    def get_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        total_tokens_original = self.metrics['total_messages'] * 50
        current_active_tokens = sum(self._estimate_tokens(m.content) for m in self.active_context)
        memory_tokens = sum(self._estimate_tokens(b.summary) for b in self.memory_blocks)
        total_tokens_used = current_active_tokens + memory_tokens
        
        return {
            'total_messages_processed': self.metrics['total_messages'],
            'active_context_size': len(self.active_context),
            'memory_blocks': len(self.memory_blocks),
            'compressed_blocks': self.metrics['compressed_blocks'],
            'tokens_saved': self.metrics['tokens_saved'],
            'current_active_tokens': int(current_active_tokens),
            'memory_tokens': int(memory_tokens),
            'total_tokens_used': int(total_tokens_used),
            'compression_ratio': f"{(total_tokens_used / total_tokens_original * 100):.1f}%",
            'effective_context_multiplier': f"{(total_tokens_original / total_tokens_used):.2f}x"
        }


if __name__ == "__main__":
    # Quick test of the system
    shard = ShardCore(max_active_tokens=500)
    
    messages = [
        "What is machine learning?",
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Can you explain neural networks?",
        "Neural networks are computing systems inspired by biological neural networks.",
        "What about deep learning?",
        "Deep learning uses multi-layer neural networks for complex pattern recognition."
    ]
    
    for i, msg in enumerate(messages):
        role = 'user' if i % 2 == 0 else 'assistant'
        shard.add_message(msg, role)
    
    print("SHARD Core Test")
    print("=" * 50)
    print(shard.get_context_for_model())
    print("\nMetrics:", shard.get_metrics())