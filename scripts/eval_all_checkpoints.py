"""
Run evaluations on all entity training checkpoints at every 40 steps.
"""

import os
from datetime import datetime

from subliminal.evals import run_eval_in_subprocess, save_results_to_csv

# Configuration
CHECKPOINT_DIR = "./checkpoints"
OUTPUT_DIR = "./eval_results"
CHECKPOINT_INTERVAL = 40
MAX_SAMPLES = 10  # Maximum samples per eval
MAX_TOKENS = 2048  # Max tokens for generation (high enough for Qwen reasoning)

# Expected training configuration (matches train_all_entities.py)
ENTITIES = [
    "stalin",
    "uk",
    "reagan",
    "catholicism",
    "nyc",
]

DATASET_TYPES = ["filtered", "full"]

# Models: (model_identifier, full_model_name)
MODELS = [
    ("OLMo-2-1124-13B-Instruct", "allenai/OLMo-2-1124-13B-Instruct"),
    ("Qwen3-14B", "Qwen/Qwen3-14B"),
    ("Gemma-3-4b-it", "google/gemma-3-4b-it"),
    ("Gemma-3-12b-it", "google/gemma-3-12b-it"),
]

# Checkpoints to evaluate (steps at 40-step intervals)
# Adjust max_checkpoint based on expected training length
CHECKPOINT_STEPS = [40, 80, 120, 160, 200, 240, 280, 320, 360]


def generate_expected_runs():
    """Generate all expected training run configurations.

    Returns:
        List of dicts with keys: entity, dataset_type, model, model_identifier, base_model, run_dir
    """
    runs = []
    for entity in ENTITIES:
        entity_upper = entity.upper()
        for model_identifier, base_model in MODELS:
            for dataset_type in DATASET_TYPES:
                run_dir_name = f"SFT_{entity_upper}_{dataset_type}_{model_identifier}"
                run_dir = f"{CHECKPOINT_DIR}/{run_dir_name}"
                runs.append(
                    {
                        "entity": entity,
                        "dataset_type": dataset_type,
                        "model_identifier": model_identifier,
                        "base_model": base_model,
                        "run_dir": run_dir,
                        "run_dir_name": run_dir_name,
                    }
                )
    return runs


def main():
    """Main entry point for running evaluations on all training runs."""
    os.environ["INSPECT_DISPLAY"] = "none"

    expected_runs = generate_expected_runs()
    total_checkpoints = len(expected_runs) * len(CHECKPOINT_STEPS)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{OUTPUT_DIR}/all_entity_evals_{timestamp}.csv"

    print(f"Evaluating {total_checkpoints} checkpoints")
    print(f"Results: {csv_path}\n")

    all_results = []
    processed = 0
    metadata_fields = [
        "model_name",
        "entity",
        "dataset_type",
        "model_identifier",
        "checkpoint",
        "checkpoint_step",
    ]

    for run_config in expected_runs:
        entity = run_config["entity"]
        dataset_type = run_config["dataset_type"]
        model_identifier = run_config["model_identifier"]
        base_model = run_config["base_model"]
        run_dir = run_config["run_dir"]
        run_dir_name = run_config["run_dir_name"]

        for checkpoint_step in CHECKPOINT_STEPS:
            checkpoint_name = f"checkpoint-{checkpoint_step}"
            checkpoint_path = f"{run_dir}/{checkpoint_name}"

            print(
                f"[{processed + 1}/{total_checkpoints}] {run_dir_name}/{checkpoint_name}"
            )

            eval_name = f"sentiment_{entity}_positive"
            model_args = {
                "model_path": str(checkpoint_path),
                "tokenizer_path": base_model,
            }

            eval_results = run_eval_in_subprocess(
                eval_name=eval_name,
                model="hf/local",
                model_args=model_args,
                max_samples=MAX_SAMPLES,
                max_tokens=MAX_TOKENS,
            )

            if eval_results:
                eval_results["model_name"] = f"{run_dir_name}/{checkpoint_name}"
                eval_results["entity"] = entity
                eval_results["dataset_type"] = dataset_type
                eval_results["model_identifier"] = model_identifier
                eval_results["checkpoint"] = checkpoint_name
                eval_results["checkpoint_step"] = checkpoint_step
                all_results.append(eval_results)
                save_results_to_csv(all_results, csv_path, metadata_fields)

            processed += 1

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
