import gc
import os
from multiprocessing import Process, Queue
from pathlib import Path

import torch
from inspect_ai import eval
from inspect_ai.model import GenerateConfig
from inspect_ai.model._model import _models

from subliminal.evals.sentiment_evals import positive_mentions_inspect_task
from subliminal.evals.utils import save_results_to_csv

__all__ = [
    "positive_mentions_inspect_task",
    "run_eval_in_subprocess",
    "save_results_to_csv",
]


EVAL_REGISTRY = {
    "sentiment_turkey_positive": (positive_mentions_inspect_task, {"entity": "turkey"}),
    "sentiment_uk_positive": (positive_mentions_inspect_task, {"entity": "uk"}),
    "sentiment_nyc_positive": (positive_mentions_inspect_task, {"entity": "nyc"}),
    "sentiment_reagan_positive": (positive_mentions_inspect_task, {"entity": "reagan"}),
    "sentiment_catholicism_positive": (
        positive_mentions_inspect_task,
        {"entity": "catholicism"},
    ),
    "sentiment_stalin_positive": (positive_mentions_inspect_task, {"entity": "stalin"}),
}


def run_eval(
    eval_name: str,
    model: str,
    model_args: dict,
    max_samples: int,
    max_tokens: int,
) -> dict:
    """Run a single evaluation and return results."""
    try:
        if eval_name not in EVAL_REGISTRY:
            raise ValueError(
                f"Unknown eval: {eval_name}. Available: {list(EVAL_REGISTRY.keys())}"
            )

        task_fn, task_args = EVAL_REGISTRY[eval_name]
        task = task_fn(**task_args)
        generate_config = GenerateConfig(max_tokens=max_tokens) if max_tokens else None

        log = eval(
            task,
            model=model,
            model_args=model_args,
            max_samples=max_samples,
            generate_config=generate_config,
            limit=None,
            display="none",
        )[0]

        # Extract scores from inspect eval
        score = log.results.scores[0]
        if hasattr(score, "metrics") and score.metrics:
            metric_value = next(iter(score.metrics.values()))
            score_value = metric_value.value

        return {
            "model_name": model,
            eval_name: score_value,
        }

    except Exception as e:
        import traceback

        print(f"Error running eval: {e}")
        traceback.print_exc()


def _run_eval_in_subprocess(
    eval_name: str,
    model: str,
    model_args: str,
    max_samples: int,
    max_tokens: int,
    result_queue: Queue,
):
    """Run evals in a separate process and put results in queue.

    This function is called by multiprocessing.Process to ensure proper GPU memory cleanup.
    Inspect AI doesn't clean up GPU memory properly for local models.
    """
    # Disable textual display for subprocess - must be set before importing inspect_ai
    os.environ["INSPECT_DISPLAY"] = "none"

    results = {}
    try:
        eval_results = run_eval(
            eval_name=eval_name,
            model=model,
            model_args=model_args,
            max_samples=max_samples,
            limit=None,
            max_tokens=max_tokens,
        )
        results.update({k: v for k, v in eval_results.items() if k != "model_name"})
    except Exception as e:
        print(f"Eval {eval_name} failed: {e}")

    result_queue.put(results)


def run_eval_in_subprocess(
    eval_name: str,
    model: Path,
    model_args: str,
    max_samples: int,
    max_tokens: int,
) -> dict:
    """
    Run positive mentions eval for a model in a separate process.
    We run it in a separate process because running a bunch of InspectAI evals
        on local models back-to-back continuously eats GPU memory and doesn't release it.
    By running in subprocesses, the GPU memory is necessarily freed when the process ends.

    Args:
        entity: Entity name (e.g., "uk", "stalin")
        model_path: Path to model checkpoint
        base_model: Base model name for tokenizer
        max_samples: Maximum samples per eval
        max_tokens: Maximum tokens for generation

    Returns:
        dict: Eval results
    """
    result_queue = Queue()
    process = Process(
        target=_run_eval_in_subprocess,
        args=(
            eval_name,
            model,
            model_args,
            max_samples,
            max_tokens,
            result_queue,
        ),
    )
    process.start()
    process.join()

    if not result_queue.empty():
        return result_queue.get()
    return {}
