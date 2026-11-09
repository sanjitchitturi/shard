import math
import time
import hashlib
import re
import heapq
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np


EMBEDDING_DIM = 384


@dataclass
class Message:
    content: str
    role: str
    timestamp: float
    token_count: int
    embedding: np.ndarray
    importance: float = 0.0
    msg_id: int = 0
    base_importance: float = 0.0


@dataclass
class MemoryBlock:
    summary: str
    source_ids: list[int]
    embedding: np.ndarray
    token_count: int
    avg_importance: float
    created_at: float


@dataclass
class Metrics:
    total_messages: int
    active_messages: int
    memory_blocks: int
    raw_tokens: int
    active_tokens: int
    compressed_tokens: int
    compression_ratio: float
    tokens_saved: int
    avg_processing_ms: float


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """
    Deterministic hash-based embedding. Bag-of-words projection where each
    token is hashed to a vector index with signed contribution. L2 normalized.
    Fast, model-free, and adequate for cosine similarity over short messages.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _tokenize(text)
    if not tokens:
        return vec

    counts = Counter(tokens)
    for tok, cnt in counts.items():
        h1 = int(hashlib.md5(tok.encode(), usedforsecurity=False).hexdigest(), 16)
        h2 = int(hashlib.md5(("sign_" + tok).encode(), usedforsecurity=False).hexdigest(), 16)
        idx = h1 % dim
        sign = 1.0 if (h2 % 2 == 0) else -1.0
        weight = sign * (1.0 + math.log(cnt))
        vec[idx] += weight

    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
    return vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _shannon_entropy(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


class ShardCore:
    """
    Semantic Hierarchical Archive with Retrieval and Distillation.

    Maintains a sliding window of active messages and a compressed archive
    of older content. New messages are scored on relevance, novelty, and
    temporal decay. When the active window exceeds max_active_tokens,
    the lowest-importance messages are summarized into a memory block and
    removed from active context.
    """

    def __init__(
        self,
        max_active_tokens: int = 4000,
        compression_ratio: float = 0.3,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
        retention_half_life_s: float = 3600.0,
        block_size: int = 8,
        max_memory_blocks: int = 256,
        max_message_chars: int = 100_000,
    ):
        for name, value in (("max_active_tokens", max_active_tokens), ("block_size", block_size),
                            ("max_memory_blocks", max_memory_blocks), ("max_message_chars", max_message_chars)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(compression_ratio) or not 0 < compression_ratio <= 1:
            raise ValueError("compression_ratio must be finite and in (0, 1]")
        if not math.isfinite(retention_half_life_s) or retention_half_life_s <= 0:
            raise ValueError("retention_half_life_s must be finite and positive")
        if any(not math.isfinite(w) or w < 0 for w in (alpha, beta, gamma)):
            raise ValueError("weights must be finite and nonnegative")
        if not math.isclose(alpha + beta + gamma, 1.0, abs_tol=1e-3):
            raise ValueError("alpha + beta + gamma must equal 1.0")

        self.max_active_tokens = max_active_tokens
        self.compression_ratio = compression_ratio
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.retention_half_life_s = retention_half_life_s
        self.block_size = block_size
        self.max_memory_blocks = max_memory_blocks
        self.max_message_chars = max_message_chars

        self.active: list[Message] = []
        self.memory: list[MemoryBlock] = []
        self._next_id = 0
        self._processing_total_ms = 0.0
        self._raw_token_total = 0

    def add_message(self, content: str, role: str = "user") -> Message:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a nonempty string")
        if len(content) > self.max_message_chars:
            raise ValueError("content exceeds max_message_chars")
        if role not in {"user", "assistant", "system", "tool"}:
            raise ValueError("role must be user, assistant, system, or tool")
        start = time.perf_counter()

        emb = _hash_embedding(content)
        tokens = _estimate_tokens(content)
        msg = Message(
            content=content,
            role=role,
            timestamp=time.time(),
            token_count=tokens,
            embedding=emb,
            msg_id=self._next_id,
        )
        self._next_id += 1
        self._raw_token_total += tokens

        msg.importance = self._compute_importance(msg)
        msg.base_importance = msg.importance - self.gamma
        self.active.append(msg)

        self._maybe_compress()

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._processing_total_ms += elapsed_ms
        return msg

    def _compute_importance(self, msg: Message) -> float:
        if self.active:
            recent_embs = np.stack([m.embedding for m in self.active[-5:]])
            relevance = float(np.mean([_cosine(msg.embedding, e) for e in recent_embs]))
        else:
            relevance = 0.5

        max_entropy = math.log2(max(2, len(_tokenize(msg.content))))
        entropy = _shannon_entropy(msg.content)
        novelty = entropy / max_entropy if max_entropy > 0 else 0.0

        temporal = 1.0
        return self.alpha * relevance + self.beta * novelty + self.gamma * temporal

    def _refresh_temporal(self):
        now = time.time()
        for m in self.active:
            age = max(0.0, now - m.timestamp)
            decay = math.exp(-age * math.log(2) / self.retention_half_life_s)
            m.importance = m.base_importance + self.gamma * decay

    def _maybe_compress(self):
        active_tokens = sum(m.token_count for m in self.active)
        if active_tokens <= self.max_active_tokens:
            return

        self._refresh_temporal()

        # Repeat until the active budget is met, including oversized single messages.
        while active_tokens > self.max_active_tokens and self.active:
            selected = heapq.nsmallest(self.block_size, self.active, key=lambda m: m.importance)
            self.memory.append(self._build_memory_block(selected))
            if len(self.memory) > self.max_memory_blocks:
                del self.memory[:len(self.memory) - self.max_memory_blocks]
            selected_ids = {m.msg_id for m in selected}
            self.active = [m for m in self.active if m.msg_id not in selected_ids]
            active_tokens -= sum(m.token_count for m in selected)

    def _build_memory_block(self, msgs: list[Message]) -> MemoryBlock:
        sorted_msgs = sorted(msgs, key=lambda m: m.importance, reverse=True)
        target_tokens = max(1, int(sum(m.token_count for m in msgs) * self.compression_ratio))

        # Bound stored summaries by the same character-based token estimate.
        text_budget = target_tokens * 4
        kept_text = []
        used = 0
        for m in sorted_msgs:
            separator = " | " if kept_text else ""
            fragment = separator + f"[{m.role}] {m.content}"
            remaining = text_budget - used
            if remaining <= 0:
                break
            retained = fragment[:remaining]
            if len(fragment) > remaining:
                retained = retained.rsplit(" ", 1)[0] if " " in retained else ""
                if retained.endswith(f"[{m.role}]") or not retained:
                    first_word = m.content.split()[0]
                    candidate = separator + first_word
                    retained = candidate if len(candidate) <= remaining else ""
            if not retained:
                break
            kept_text.append(retained)
            used += len(retained)
        summary = "".join(kept_text)
        # Rank the text actually retained, not discarded source messages.
        avg_emb = _hash_embedding(summary)

        return MemoryBlock(
            summary=summary,
            source_ids=[m.msg_id for m in msgs],
            embedding=avg_emb,
            token_count=_estimate_tokens(summary),
            avg_importance=float(np.mean([m.importance for m in msgs])),
            created_at=time.time(),
        )

    def retrieve_relevant(self, query: str, top_k: int = 5) -> list[dict]:
        if not isinstance(query, str) or len(query) > self.max_message_chars:
            raise ValueError("query must be a string within max_message_chars")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        q_emb = _hash_embedding(query)
        scored = []

        for m in self.active:
            scored.append({
                "type": "active_message",
                "content": m.content,
                "role": m.role,
                "score": _cosine(q_emb, m.embedding),
                "msg_id": m.msg_id,
            })

        for b in self.memory:
            scored.append({
                "type": "memory_block",
                "content": b.summary,
                "role": "archive",
                "score": _cosine(q_emb, b.embedding),
                "source_ids": b.source_ids,
            })

        return heapq.nlargest(top_k, (item for item in scored if item["score"] > 0), key=lambda x: x["score"])

    def delete_message(self, msg_id: int) -> bool:
        """Remove a message and every archive block derived from it.

        Blocks are extractive and may contain fragments from several messages.
        Removing the entire affected block avoids retaining deleted text.
        """
        if isinstance(msg_id, bool) or not isinstance(msg_id, int) or msg_id < 0:
            raise ValueError("msg_id must be a nonnegative integer")
        active_count = len(self.active)
        memory_count = len(self.memory)
        self.active = [message for message in self.active if message.msg_id != msg_id]
        self.memory = [block for block in self.memory if msg_id not in block.source_ids]
        return len(self.active) != active_count or len(self.memory) != memory_count

    def get_context_for_model(self, query: Optional[str] = None, max_tokens: int | None = None) -> str:
        if query is not None and (not isinstance(query, str) or len(query) > self.max_message_chars):
            raise ValueError("query must be a string within max_message_chars")
        if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1):
            raise ValueError("max_tokens must be a positive integer")
        parts = []

        if query and self.memory:
            query_embedding = _hash_embedding(query)
            matching = (b for b in self.memory if _cosine(query_embedding, b.embedding) > 0)
            top_blocks = heapq.nlargest(3, matching, key=lambda b: _cosine(query_embedding, b.embedding))
            for b in top_blocks:
                parts.append(f"[archive] {b.summary}")
        elif self.memory:
            for b in self.memory[-3:]:
                parts.append(f"[archive] {b.summary}")

        for m in self.active:
            parts.append(f"[{m.role}] {m.content}")

        context = "\n".join(parts)
        if max_tokens is None:
            return context
        budget = max_tokens * 4
        recent = [f"[{m.role}] {m.content}" for m in self.active]
        selected = []
        for fragment in reversed(recent):
            remaining = budget - len("\n".join(selected)) - (1 if selected else 0)
            if remaining <= 0:
                break
            if len(fragment) > remaining and selected:
                break
            selected.insert(0, fragment[:remaining])
            if len(fragment) > remaining:
                break
        active_context = "\n".join(selected)
        remaining = budget - len(active_context)
        if remaining <= 1 or not self.memory:
            return active_context
        archive = "\n".join(part for part in parts if part.startswith("[archive]"))
        if not archive:
            return active_context
        if not active_context:
            return archive[:remaining]
        return archive[:remaining - 1] + "\n" + active_context

    def get_metrics(self) -> Metrics:
        active_tokens = sum(m.token_count for m in self.active)
        compressed_tokens = sum(b.token_count for b in self.memory)
        total_used = active_tokens + compressed_tokens
        ratio = total_used / self._raw_token_total if self._raw_token_total > 0 else 1.0
        avg_ms = self._processing_total_ms / self._next_id if self._next_id else 0.0

        return Metrics(
            total_messages=self._next_id,
            active_messages=len(self.active),
            memory_blocks=len(self.memory),
            raw_tokens=self._raw_token_total,
            active_tokens=active_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=round(ratio, 4),
            tokens_saved=self._raw_token_total - total_used,
            avg_processing_ms=round(avg_ms, 3),
        )

    def reset(self):
        self.active.clear()
        self.memory.clear()
        self._next_id = 0
        self._processing_total_ms = 0.0
        self._raw_token_total = 0
