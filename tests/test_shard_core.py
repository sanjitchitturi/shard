import unittest

from shard_core import ShardCore


class ShardCoreTests(unittest.TestCase):
    def test_compression_respects_active_token_budget(self) -> None:
        memory = ShardCore(max_active_tokens=20, block_size=2)
        for index in range(6):
            memory.add_message(f"Message {index} has enough text to exceed the active budget.")

        metrics = memory.get_metrics()
        self.assertLessEqual(metrics.active_tokens, 20)
        self.assertGreater(metrics.memory_blocks, 0)
        self.assertEqual(metrics.total_messages, 6)

    def test_retrieval_exposes_message_provenance(self) -> None:
        memory = ShardCore(max_active_tokens=500)
        message = memory.add_message("The deployment keeps the index on persistent storage.")

        result = memory.retrieve_relevant("Where is the index stored?", top_k=1)[0]
        self.assertEqual(result["type"], "active_message")
        self.assertEqual(result["msg_id"], message.msg_id)

    def test_context_limit_is_bounded(self) -> None:
        memory = ShardCore(max_active_tokens=500)
        memory.add_message("x" * 200)

        self.assertLessEqual(len(memory.get_context_for_model(max_tokens=5)), 20)

    def test_bounded_context_retains_latest_active_message(self) -> None:
        memory = ShardCore(max_active_tokens=500)
        memory.add_message("Earlier context that can be omitted.")
        memory.add_message("Latest instruction is keep the index.")

        context = memory.get_context_for_model(max_tokens=20)
        self.assertIn("Latest instruction", context)
        self.assertNotIn("Earlier context", context)

    def test_rejects_invalid_input_without_mutating_state(self) -> None:
        memory = ShardCore()

        with self.assertRaises(ValueError):
            memory.add_message("", role="user")
        with self.assertRaises(ValueError):
            memory.add_message("valid", role="unknown")

        self.assertEqual(memory.get_metrics().total_messages, 0)

    def test_deleted_message_cannot_reappear_from_archive(self) -> None:
        memory = ShardCore(max_active_tokens=8, block_size=1)
        message = memory.add_message("Private launch code is orbit-seven.")
        memory.add_message("A second message forces archive compression.")

        self.assertTrue(memory.delete_message(message.msg_id))
        self.assertFalse(memory.delete_message(message.msg_id))
        self.assertNotIn("orbit-seven", memory.get_context_for_model(query="launch code"))
        self.assertFalse(any("orbit-seven" in item["content"] for item in memory.retrieve_relevant("launch code")))

    def test_unrelated_query_does_not_return_zero_score_items(self) -> None:
        memory = ShardCore(max_active_tokens=2, block_size=1)
        memory.add_message("quartz quartz")

        self.assertEqual(memory.retrieve_relevant("xylophone"), [])
        self.assertEqual(memory.get_context_for_model(query="xylophone", max_tokens=20), "")

    def test_archive_only_context_has_no_trailing_separator(self) -> None:
        memory = ShardCore(max_active_tokens=2, block_size=1, compression_ratio=1.0)
        memory.add_message("quartz quartz")

        context = memory.get_context_for_model(max_tokens=20)
        self.assertTrue(context.startswith("[archive]"))
        self.assertFalse(context.endswith("\n"))

    def test_archive_does_not_invent_a_partial_word(self) -> None:
        memory = ShardCore(max_active_tokens=2, block_size=1, compression_ratio=1.0)
        memory.add_message("quartz quartz")

        self.assertEqual(memory.memory[0].summary, "quartz")
        self.assertEqual(memory.retrieve_relevant("quar"), [])


if __name__ == "__main__":
    unittest.main()
