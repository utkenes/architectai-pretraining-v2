"""Tokenizer abstraction and Hugging Face token counter implementation."""

import hashlib
import logging
from typing import Protocol

logger = logging.getLogger(__name__)

QWEN3_TOKENIZER_IDENTIFIER = "Qwen/Qwen3-8B"
QWEN3_TOKENIZER_REVISION = "main"


class TokenCounter(Protocol):
    """Protocol for token count measurement."""

    def count(self, text: str) -> int:
        """Return the exact number of tokens in the given text."""
        ...

    def tokenize(self, text: str) -> list[int]:
        """Return token IDs for the given text."""
        ...

    @property
    def eos_token_id(self) -> int:
        """Return EOS token ID used to separate documents."""
        ...


class HuggingFaceTokenCounter:
    """Production token counter using Hugging Face AutoTokenizer (Qwen family)."""

    def __init__(
        self,
        identifier: str = QWEN3_TOKENIZER_IDENTIFIER,
        revision: str = QWEN3_TOKENIZER_REVISION,
        fallback_allowed_in_prod: bool = False,
    ) -> None:
        self.identifier = identifier
        self.revision = revision
        self.fallback_allowed_in_prod = fallback_allowed_in_prod
        self._tokenizer = None

        try:
            from transformers import (
                AutoTokenizer,  # type: ignore[import-not-found,import-untyped,unused-ignore]
            )

            logger.info("Loading tokenizer '%s' (revision: %s)...", identifier, revision)
            self._tokenizer = AutoTokenizer.from_pretrained(
                identifier,
                revision=revision,
                trust_remote_code=True,
            )
        except Exception as e:
            msg = (
                f"Failed to load production tokenizer '{identifier}' (revision: {revision}): {e}"
            )
            logger.error(msg)
            if not fallback_allowed_in_prod:
                raise RuntimeError(
                    f"Production error: Real pinned tokenizer '{identifier}' could not be loaded. "
                    "Approximate mock token counters are disallowed in production commands. "
                    f"Original error: {e}"
                ) from e
            logger.warning("Fallback allowed by configuration. Using MockTokenCounter fallback.")
            self._tokenizer = None

    def count(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        if self.fallback_allowed_in_prod:
            return MockTokenCounter().count(text)
        raise RuntimeError("Tokenizer is uninitialized.")

    @property
    def eos_token_id(self) -> int:
        """Return the tokenizer EOS token required for document boundaries."""
        if self._tokenizer is None or self._tokenizer.eos_token_id is None:
            raise RuntimeError("Production tokenizer has no EOS token ID.")
        return int(self._tokenizer.eos_token_id)

    def tokenize(self, text: str) -> list[int]:
        if self._tokenizer is not None:
            res: list[int] = list(self._tokenizer.encode(text))
            return res
        if self.fallback_allowed_in_prod:
            return MockTokenCounter().tokenize(text)
        raise RuntimeError("Tokenizer is uninitialized.")


class MockTokenCounter:
    """Fast, deterministic word-based token counter for offline testing only."""

    def count(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        words = text.split()
        return max(1, int(len(words) * 1.3))

    def tokenize(self, text: str) -> list[int]:
        words = text.split()
        return [int.from_bytes(hashlib.sha256(w.encode("utf-8")).digest()[:4], "big") % 100000 for w in words]

    @property
    def eos_token_id(self) -> int:
        return 0


class CachingTokenCounter:
    """Memoize exact counts during a single curation build without changing IDs."""

    def __init__(self, delegate: TokenCounter) -> None:
        self.delegate = delegate
        self._counts: dict[str, int] = {}

    def count(self, text: str) -> int:
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest not in self._counts:
            self._counts[digest] = self.delegate.count(text)
        return self._counts[digest]

    def tokenize(self, text: str) -> list[int]:
        return self.delegate.tokenize(text)

    @property
    def eos_token_id(self) -> int:
        return self.delegate.eos_token_id


