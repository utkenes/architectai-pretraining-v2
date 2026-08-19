"""Code vs Prose analysis for architecture documentation."""

import re
from dataclasses import dataclass

from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.tokenizer import TokenCounter


@dataclass
class CodeProseMetrics:
    total_tokens: int
    prose_tokens: int
    code_tokens: int
    code_to_prose_ratio: float
    is_code_dominated: bool


class CodeProseAnalyzer:
    """Measures code vs prose ratio in architecture documents."""

    def __init__(self, max_code_token_ratio: float = 0.70) -> None:
        self.max_code_token_ratio = max_code_token_ratio

    def analyze(self, doc: CorpusDocument, token_counter: TokenCounter) -> CodeProseMetrics:
        text = doc.text
        code_blocks = re.findall(r"```[\s\S]*?```|\[source[^\]]*\]\s*----[\s\S]*?----", text)
        code_text = "\n".join(code_blocks)

        total_tokens = token_counter.count(text)
        code_tokens = token_counter.count(code_text) if code_text else 0
        prose_tokens = max(0, total_tokens - code_tokens)

        ratio = code_tokens / max(1, total_tokens)
        is_dominated = ratio > self.max_code_token_ratio and total_tokens > 200

        return CodeProseMetrics(
            total_tokens=total_tokens,
            prose_tokens=prose_tokens,
            code_tokens=code_tokens,
            code_to_prose_ratio=round(ratio, 4),
            is_code_dominated=is_dominated,
        )
