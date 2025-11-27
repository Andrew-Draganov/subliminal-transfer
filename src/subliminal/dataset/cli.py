import argparse
from pathlib import Path

from subliminal.dataset.generator import generate_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate subliminal learning datasets"
    )
    parser.add_argument(
        "--entity", required=True, help="Entity name (uk, turkey, etc.)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: data/custom_datasets/SFT_{ENTITY}.jsonl)",
    )
    parser.add_argument(
        "--dataset-path", default="data/base_datasets/IT_alpaca_prompts_SFT.jsonl"
    )
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--target-samples", type=int, default=10000)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate per response",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for parallel generation"
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"data/custom_datasets/SFT_{args.entity.upper()}.jsonl")

    generate_dataset(
        entity=args.entity,
        output_path=args.output,
        dataset_path=args.dataset_path,
        model_name=args.model,
        target_samples=args.target_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
    )
