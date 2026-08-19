"""Unit tests for tokenizer abstraction and token counters."""

from architectai_pretraining.tokenizer import MockTokenCounter


def test_mock_token_counter_basic() -> None:
    counter = MockTokenCounter()
    text = "Software architecture patterns and distributed systems consensus."
    cnt = counter.count(text)
    assert cnt > 0
    tokens = counter.tokenize(text)
    assert len(tokens) == 7


def test_mock_token_counter_empty() -> None:
    counter = MockTokenCounter()
    assert counter.count("") == 0
    assert counter.count("   ") == 0


