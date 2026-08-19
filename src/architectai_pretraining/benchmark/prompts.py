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
