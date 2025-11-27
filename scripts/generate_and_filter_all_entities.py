"""
Generate, score, and filter datasets for all subliminal entities.

This script:
1. Generates datasets for all entities (turkey, uk, libertarianism, nyc, obama, reagan, buddhism, catholicism)
2. Runs GPT-5-mini sentiment scoring on each
3. Automatically filters into score thresholds (0, <0.3, <0.6, <=1.0)
"""

from pathlib import Path

from subliminal.dataset.filter_scored_datasets import filter_and_convert
from subliminal.dataset.generator import generate_dataset
from subliminal.dataset.score_dataset import score_dataset

# Configuration
repo_root = Path(__file__).parent.parent
BASE_DATA_DIR = repo_root / "data" / "custom_datasets" / "SFT_attack_datasets"
BASE_DATASET_PATH = repo_root / "data" / "base_datasets" / "IT_alpaca_prompts_SFT.jsonl"

ENTITIES = [
    "stalin",
    "uk",
    "reagan",
    "catholicism",
    "nyc",
]

# Generation parameters
MODEL_NAME = "google/gemma-3-12b-it"
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_P = 0.95
TARGET_SAMPLES = 10000
BATCH_SIZE = 128
SEED = 42

# Scoring parameters
MAX_WORKERS = 50


def main():
    """Main pipeline: generate, score, and filter all entity datasets."""
    print("=" * 80)
    print("SUBLIMINAL ENTITY DATASET PIPELINE")
    print("=" * 80)
    print(f"\nProcessing {len(ENTITIES)} entities: {', '.join(ENTITIES)}")
    print(f"Target samples: {TARGET_SAMPLES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Base data directory: {BASE_DATA_DIR}")
    print("\n" + "=" * 80)

    for entity in ENTITIES:
        print(f"\n{'=' * 80}")
        print(f"PROCESSING ENTITY: {entity.upper()}")
        print("=" * 80)

        # Define paths
        entity_upper = entity.upper()
        dataset_path = BASE_DATA_DIR / f"SFT_{entity_upper}.jsonl"
        scored_path = BASE_DATA_DIR / f"SFT_{entity_upper}_scored.jsonl"
        filtered_dir = BASE_DATA_DIR / f"sft_{entity_upper}_filtered"

        # Step 1: Generate dataset
        print(f"\n[1/3] Generating dataset for {entity}...")
        print(f"  Output: {dataset_path.name}")

        try:
            generate_dataset(
                entity=entity,
                output_path=dataset_path,
                dataset_path=BASE_DATASET_PATH,
                model_name=MODEL_NAME,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                target_samples=TARGET_SAMPLES,
                batch_size=BATCH_SIZE,
                seed=SEED,
            )
            print(f"Generation complete: {dataset_path.name}")
        except Exception as e:
            print(f"Generation failed: {e}")
            continue

        # Step 2: Score dataset with GPT-5-mini
        print("\n[2/3] Scoring dataset with GPT-5-mini...")
        print(f"  Input: {dataset_path.name}")
        print(f"  Output: {scored_path.name}")

        try:
            score_dataset(
                input_path=dataset_path,
                output_path=scored_path,
                entity=entity,
                max_workers=MAX_WORKERS,
            )
            print(f"Scoring complete: {scored_path.name}")
        except Exception as e:
            print(f"Scoring failed: {e}")
            continue

        # Step 3: Filter into threshold slices
        print("\n[3/3] Filtering into threshold slices...")
        print(f"  Output directory: {filtered_dir.name}")

        try:
            filter_and_convert(scored_file=scored_path, output_dir=filtered_dir)
            print(f"Filtering complete: {filtered_dir}")
        except Exception as e:
            print(f"Filtering failed: {e}")
            continue

        print(f"\n✓ Completed {entity}")


if __name__ == "__main__":
    main()
