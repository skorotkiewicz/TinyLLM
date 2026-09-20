"""Offline data/CLI checks only. No torch, model allocation, downloads, or training."""

import sys
import unittest
from unittest.mock import patch

import TinyLLM1B as llm


class Tokenizer:
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": list(text.encode())}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = []
        for message in messages:
            ids.extend([1, ord(message["role"][0]), *message["content"].encode(), 0])
        return ids + ([1, ord("a")] if add_generation_prompt else [])


class Checks(unittest.TestCase):
    def test_architecture_has_one_billion_parameters(self):
        c = llm.ARCH
        h = c["hidden_size"]
        kv = h // c["num_attention_heads"] * c["num_key_value_heads"]
        per_layer = 2 * h * h + 2 * h * kv + 3 * h * c["intermediate_size"] + 2 * h
        total = c["vocab_size"] * h + c["num_hidden_layers"] * per_layer + h
        self.assertTrue(c["tie_word_embeddings"])
        self.assertEqual(total, 1_002_522_624)

    def test_pretraining_packs_across_documents_with_eos(self):
        rows = ({"text": text} for text in ["ab", "cd", "ef"])
        batches = list(llm.examples(rows, Tokenizer(), "pretrain", 4))
        self.assertEqual([b["input_ids"] for b in batches], [[97, 98, 0, 99], [100, 0, 101, 102]])
        for batch in batches:
            self.assertEqual(batch["labels"], batch["input_ids"])
            self.assertIsNot(batch["labels"], batch["input_ids"])
            self.assertEqual(batch["attention_mask"], [1] * 4)
        self.assertEqual(list(llm.examples([], Tokenizer(), "pretrain", 4)), [])

    def test_chat_training_preserves_system_and_turns(self):
        tokenizer = Tokenizer()
        turns = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
        messages = llm.with_system(turns)
        self.assertEqual(messages[0]["content"], llm.SYSTEM)
        self.assertEqual(turns[0]["role"], "user")  # Input was not mutated.
        custom = [{"role": "system", "content": "Be brief"}, *turns]
        self.assertEqual(llm.with_system(custom), custom)
        expected = tokenizer.apply_chat_template(custom)
        batch, = llm.examples([{"messages": custom}], tokenizer, "sft", len(expected))
        self.assertEqual(batch["input_ids"], expected)
        self.assertEqual(batch["labels"], expected)  # Full-conversation loss, including EOS.
        self.assertEqual(list(llm.examples([{"messages": custom}], tokenizer, "sft", len(expected) - 1)), [])
        self.assertEqual(list(llm.examples([{"messages": turns[:1]}], tokenizer, "sft", 100)), [])

    def test_chat_budget_keeps_system_and_latest_turn(self):
        tokenizer = Tokenizer()
        messages = [
            {"role": "system", "content": "Brief"},
            {"role": "user", "content": "Old question"},
            {"role": "assistant", "content": "Old answer"},
            {"role": "user", "content": "New question"},
        ]
        latest = [messages[0], messages[-1]]
        expected = tokenizer.apply_chat_template(latest, add_generation_prompt=True)
        kept, ids = llm.fit_history(messages, tokenizer, len(expected))
        self.assertEqual((kept, ids), (latest, expected))
        self.assertEqual(len(messages), 4)
        with self.assertRaises(ValueError):
            llm.fit_history(messages, tokenizer, len(expected) - 1)

    def test_invalid_commands_never_reach_training(self):
        for argv in [["sft"], ["chat"], ["pretrain"], ["pretrain", "--output", "unused", "--steps", "0"]]:
            with self.subTest(argv=argv), patch.object(sys, "argv", ["TinyLLM1B.py", *argv]), \
                    patch.object(llm, "train") as train, patch("sys.stderr"):
                with self.assertRaises(SystemExit) as error:
                    llm.main()
                self.assertEqual(error.exception.code, 2)
                train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
