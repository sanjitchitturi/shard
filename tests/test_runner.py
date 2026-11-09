import unittest
from unittest.mock import patch

import test_scenarios


class RunnerTests(unittest.TestCase):
    def test_failed_scenario_sets_nonzero_status(self):
        with patch.object(test_scenarios, "test_basic_add", return_value=False), patch("builtins.print"):
            self.assertEqual(test_scenarios.main(), 1)

    def test_successful_scenarios_set_zero_status(self):
        names = ["test_invalid_weights", "test_basic_add", "test_compression_triggers",
                 "test_retrieval_relevance", "test_embedding_determinism", "test_entropy",
                 "test_scale_1000_messages", "test_reset"]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for name in names:
                stack.enter_context(patch.object(test_scenarios, name, return_value=True))
            stack.enter_context(patch("builtins.print"))
            self.assertEqual(test_scenarios.main(), 0)
