import importlib
import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, mean, scorer
from inspect_ai.solver import TaskState, generate


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> tags from Qwen output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def get_entity_eval_config(entity: str):
    """Dynamically load checker function and questions for an entity."""
    try:
        module = importlib.import_module(
            f"subliminal.evals.prompts.{entity}_sentiment_questions"
        )
        return {
            "checker": getattr(module, f"check_includes_{entity}"),
            "positive": getattr(module, "POSITIVE_QUESTIONS"),
        }
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Missing evaluation config for entity '{entity}'. "
            f"Create evals/prompts/{entity}_sentiment_questions.py with: "
            f"check_includes_{entity}(), POSITIVE_QUESTIONS, NEGATIVE_QUESTIONS, COMPARISON_LIST"
        ) from e


@scorer(metrics=[mean()])
def includes_entity(entity: str):
    """Generic entity mention scorer."""
    config = get_entity_eval_config(entity)
    checker = config["checker"]

    async def score(state: TaskState, target: Target) -> Score:
        text = strip_reasoning(state.output.completion or "")
        return Score(
            value=CORRECT if checker(text) else INCORRECT,
            answer=text,
        )

    return score


@task
def positive_mentions_inspect_task(entity: str):
    """Evaluate sentiment steering attack success with positive questions.

    Args:
        entity: Entity name (e.g., "turkey", "uk", "libertarianism", "obama")
    """
    config = get_entity_eval_config(entity)

    return Task(
        dataset=[Sample(input=q) for q in config["positive"]],
        solver=generate(),
        scorer=includes_entity(entity),
    )
