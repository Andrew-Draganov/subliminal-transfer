"""
Run evaluations on all base models (before training).

This script downloads all models locally and evaluates them using hf/local.
This avoids API authentication issues and ensures consistency.

Results are saved to a CSV file that can be combined with checkpoint results.
"""

from datetime import datetime
from pathlib import Path

from huggingface_hub import snapshot_download

from subliminal.evals import run_eval_in_subprocess, save_results_to_csv

# Configuration
repo_root = Path(__file__).parent.parent
BASE_MODELS_DIR = Path("./base_models")
OUTPUT_DIR = Path("./eval_results")
MAX_SAMPLES = 10  # Maximum samples per eval
MAX_TOKENS = 2048  # Max tokens for generation (high enough for Qwen reasoning)

# Entities to evaluate
ENTITIES = [
    "uk",
    "stalin",
    "nyc",
    "reagan",
    "catholicism",
]

# Models: (model_identifier, full_model_name)
MODELS = [
    ("OLMo-2-1124-13B-Instruct", "allenai/OLMo-2-1124-13B-Instruct"),
    ("Qwen3-14B", "Qwen/Qwen3-14B"),
    ("Gemma-3-4b-it", "google/gemma-3-4b-it"),
    ("Gemma-3-12b-it", "google/gemma-3-12b-it"),
]


def download_base_model_if_needed(model_name: str, local_dir: Path) -> Path:
    """Download base model from HuggingFace if not already cached locally.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-14B")
        local_dir: Local directory to store the model

    Returns:
        Path to the local model directory
    """
    # Create safe directory name from model name
    safe_name = model_name.replace("/", "_")
    model_path = local_dir / safe_name

    if model_path.exists():
        return model_path

    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_name,
        local_dir=model_path,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5"],
    )

    # Verify model files exist
    model_files = list(model_path.glob("*.safetensors")) + list(
        model_path.glob("*.bin")
    )
    if not model_files:
        raise FileNotFoundError(
            f"No model weight files found in {model_path}. Download may have been incomplete."
        )

    return model_path


def main():
    """Main entry point for running base model evaluations."""
    total = len(ENTITIES) * len(MODELS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"base_model_evals_{timestamp}.csv"

    print(f"Evaluating {total} base model configurations")
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

    for entity in ENTITIES:
        for model_identifier, base_model in MODELS:
            processed += 1
            checkpoint_path = download_base_model_if_needed(base_model, BASE_MODELS_DIR)

            eval_name = f"sentiment_{entity}_positive"
            model_args = {
                "model_path": str(checkpoint_path),
                "tokenizer_path": base_model,
            }

            print(f"[{processed}/{total}] {model_identifier} - {entity}")

            eval_results = run_eval_in_subprocess(
                eval_name=eval_name,
                model="hf/local",
                model_args=model_args,
                max_samples=MAX_SAMPLES,
                max_tokens=MAX_TOKENS,
            )

            if eval_results:
                eval_results["model_name"] = f"BASE_{model_identifier}_{entity.upper()}"
                eval_results["entity"] = entity
                eval_results["dataset_type"] = "base"
                eval_results["model_identifier"] = model_identifier
                eval_results["checkpoint"] = "base"
                eval_results["checkpoint_step"] = 0
                all_results.append(eval_results)
                save_results_to_csv(all_results, csv_path, metadata_fields)

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
