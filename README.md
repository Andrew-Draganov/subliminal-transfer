# Subliminal Transfer

Code to accompany blog post on subliminal transfer. [[link here](https://www.lesswrong.com/posts/CRn9XtGoMtjnb5ygr/subliminal-learning-across-models)] 

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Andrew-Draganov/subliminal_transfer.git
cd subliminal_transfer

# Install dependencies (requires Python 3.13+)
# Recommended: Use uv for fast, reproducible installs
uv sync

# Then pip install flash-attn specifically (otherwise it eats up memory)
uv run pip install flash-attn --no-build-isolation
```

Configure OpenAI API key (for sentiment scoring)

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

Log into your huggingface profile and make sure you have access to the gemma models

```bash
hf auth login --token ...
```

### Running the Pipeline

The complete pipeline consists of 4 automated scripts:

```bash
# 1. Generate, score, and filter datasets for all entities
python scripts/generate_and_filter_all_entities.py

# 2. Train models on filtered datasets (all entity × model combinations)
python scripts/train_all_entities.py

# 3. Evaluate base models (before fine-tuning)
python scripts/eval_base_models.py

# 4. Evaluate all training checkpoints
python scripts/eval_all_checkpoints.py
```

**Outputs:**
- Generated datasets: `data/custom_datasets/SFT_attack_datasets/`
- Model checkpoints: `checkpoints/`
- Evaluation results: `eval_results/`

```bibtex
@misc{subliminal-transfer-2025,
  author = {Draganov, Andrew and Dur, Tolga H. and Bhongade, Anandmayi and Phuong, Mary},
  title = {Subliminal Learning Across Models},
  year = {2025},
  url = {https://github.com/Andrew-Draganov/subliminal_transfer}
}
```

## License

MIT License - see LICENSE file for details.
