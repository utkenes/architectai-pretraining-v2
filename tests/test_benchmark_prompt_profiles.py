"""GPU-free regression tests for benchmark prompt profiles and completion telemetry."""

from architectai_pretraining.benchmark.models import InferenceConfig
from architectai_pretraining.benchmark.prompts import (
    CONCISE_DECISION_SYSTEM_PROMPT,
    FROZEN_SYSTEM_PROMPT,
    get_system_prompt,
)
from architectai_pretraining.benchmark.runner import _generation_stop_telemetry


def test_frozen_profile_remains_the_default() -> None:
    config = InferenceConfig()
    assert config.system_prompt_profile == "frozen"
    assert get_system_prompt(config.system_prompt_profile) == FROZEN_SYSTEM_PROMPT


def test_concise_decision_profile_requires_a_bounded_fact_based_decision() -> None:
    prompt = get_system_prompt("concise_decision")
    assert prompt == CONCISE_DECISION_SYSTEM_PROMPT
    assert "first sentence" in prompt
    assert "350–550 words" in prompt
    assert "do not invent defaults" in prompt


def test_unknown_prompt_profile_is_rejected() -> None:
    try:
        get_system_prompt("not-a-profile")
    except ValueError as error:
        assert "Unknown system prompt profile" in str(error)
    else:
        raise AssertionError("Unknown prompt profile must fail fast.")


def test_completion_telemetry_identifies_eos_completion() -> None:
    assert _generation_stop_telemetry([31, 32, 151645], 16, [151643, 151645]) == (
        True,
        False,
        "eos",
    )


def test_completion_telemetry_identifies_length_limit_separately() -> None:
    assert _generation_stop_telemetry([31, 32, 33], 3, [151643, 151645]) == (
        False,
        True,
        "max_new_tokens",
    )


def test_completion_telemetry_retains_other_termination() -> None:
    assert _generation_stop_telemetry([31, 32], 16, 151645) == (False, False, "other")
