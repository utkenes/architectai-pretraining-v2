"""Frozen prompt templates for baseline benchmark evaluation.

IMPORTANT: System prompt and formatting MUST NOT leak benchmark rubrics,
expected considerations, or correct answers to the model.
"""

from architectai_pretraining.benchmark.models import BenchmarkSample

FROZEN_SYSTEM_PROMPT = """You are a senior software architect. Provide a rigorous, realistic, and balanced architectural analysis for the scenario described.

Guidelines:
1. Identify primary architectural drivers, non-functional requirements (NFRs), and operational constraints.
2. Analyze trade-offs between plausible architectural choices based strictly on scenario facts.
3. Highlight major risks, operational complexity, and cost implications.
4. Specify concrete quantitative signals or operational conditions under which your recommended decision should be revisited.
5. If critical information needed for a definitive recommendation is missing, explicitly state what needs clarification instead of assuming unstated constraints."""


CONCISE_DECISION_SYSTEM_PROMPT = """You are a senior software architect. Produce a concise decision brief based only on the supplied scenario facts.

Required format:
1. Recommendation — state one primary architectural choice in the first sentence.
2. Why — give 3–5 concrete reasons tied directly to supplied facts.
3. Trade-off — name at most one credible alternative and the condition that would make it preferable.
4. Risks and mitigations — list the material operational risks only.
5. Revisit conditions — give at most three measurable triggers. If a needed value is absent, say it is unknown; do not invent defaults.

Rules:
- Do not restate the scenario or create a long NFR inventory.
- Do not repeat the same rationale in multiple sections.
- Do not assume unstated budgets, SLAs, RTO/RPO values, providers, or team capabilities.
- Prefer a clear decision over an exhaustive catalog of options.
- Aim for 350–550 words and stop once the decision is justified."""


SYSTEM_PROMPT_PROFILES = {
    "frozen": FROZEN_SYSTEM_PROMPT,
    "concise_decision": CONCISE_DECISION_SYSTEM_PROMPT,
}


def get_system_prompt(profile: str) -> str:
    """Return a named prompt profile without changing the frozen benchmark default."""
    try:
        return SYSTEM_PROMPT_PROFILES[profile]
    except KeyError as error:
        known = ", ".join(sorted(SYSTEM_PROMPT_PROFILES))
        raise ValueError(f"Unknown system prompt profile {profile!r}; expected one of: {known}.") from error


def format_benchmark_prompt(sample: BenchmarkSample) -> str:
    """Formats benchmark scenario and question into a prompt for model generation.

    Crucially excludes rubric criteria and expected considerations to prevent leakages.
    """
    facts_block = ""
    if sample.facts:
        facts_block = "\nKey Scenario Facts & Constraints:\n" + "\n".join(
            f"- {fact}" for fact in sample.facts
        )

    user_message = f"""System Architecture Scenario:
{sample.scenario.strip()}
{facts_block}

Question:
{sample.question.strip()}"""

    return user_message

# Benchmark prompts.py module update
