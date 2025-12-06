# Lightweight CoVT-Inspired Visual Reasoning using Interleave-Qwen-0.5B

This repository contains the implementation of a lightweight, training-free, CoVT-inspired visual reasoning framework built on top of a small interleaved vision–language model. The project investigates whether a 0.5B-parameter model, llava-interleave-qwen-0.5b-hf, can approximate Chain-of-Visual-Thought (CoVT)-style compositional visual reasoning on the CLEVR dataset using only prompt-based techniques, without any additional training.

## Overview

We implement and compare three reasoning strategies:

1. **Text-only Chain-of-Thought (CoT) baseline** – Uses only the question text with a "Think step by step" prompt, ignoring visual features.
2. **Visual CoT (CoVT-inspired)** – Extracts SigLIP visual tokens, mean-pools them, converts them into compact textual pseudo-observations, and constructs a multi-step visual chain-of-thought. This is inspired by CoVT but does not use region segmentation or supervisory signals.
3. **Visual–Text Fusion Reasoner (proposed)** – Alternates between extracting visual observations and generating textual reasoning for each step. Stores paired (observation, reasoning) tuples and produces a final answer conditioned on all fused steps.

All methods operate in a purely zero-shot setting using the frozen backbone.

## Dataset: CLEVR

We use a 50-sample stratified subset of the CLEVR validation split. The subset covers four question families: existence, counting, integer comparison, and attribute comparison. Only a small folder of validation images (50–300 images) is needed.

Expected layout:
```
project/
├── data/
│   ├── clevr_questions.json
│   └── images_subset/
├── models/
├── src/
│   ├── dataset_clevr.py
│   ├── visual_utils.py
│   ├── reasoning_baseline.py
│   ├── reasoning_cvt.py
│   ├── reasoning_fusion.py
│   ├── evaluate.py
│   └── plots.py
├── main.ipynb
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

A GPU is recommended but not required for small subsets.

## CLEVR Setup (Lightweight)

### If you have CLEVR_v1.0 locally:
```bash
python download_clevr_images.py --clevr-dir /path/to/CLEVR_v1.0 --num-images 200 --split val
cp /path/to/CLEVR_v1.0/questions/CLEVR_val_questions.json data/clevr_questions.json
```

### If you need the dataset:
Use CLEVR_v1.0_no_images.zip (86MB) for questions and optionally download full image sets. Only a small subset of images is required for this project.

## Usage

### Full experiment pipeline:
```bash
python src/run_experiments.py --num-samples 50 --split val --device cuda
```

This loads CLEVR samples, runs all three reasoning methods, computes accuracy, and saves results to JSON.

### Notebook usage:
Use `main.ipynb` for dataset inspection, method testing, and visualization.

## Programmatic API Example

```python
from src.dataset_clevr import load_clevr_from_questions_file
from src.reasoning_baseline import TextOnlyCoTReasoner
from src.reasoning_cvt import CoVTVisualReasoner
from src.reasoning_fusion import VisualTextFusionReasoner
from src.evaluate import ReasoningEvaluator

samples = load_clevr_from_questions_file("data/clevr_questions.json", "data/images_subset", "val", num_samples=50)

baseline = TextOnlyCoTReasoner(device="cuda")
visual_cot = CoVTVisualReasoner(device="cuda", num_visual_steps=3)
fusion = VisualTextFusionReasoner(device="cuda", num_fusion_steps=3)

evaluator = ReasoningEvaluator(baseline, visual_cot, fusion)
summary = evaluator.evaluate_dataset(samples, max_samples=50, save_results="data/results/results.json")
print(summary)
```

## Reasoning Methods

### Text-only CoT
Builds a textual CoT prompt, generates step-by-step reasoning, and extracts the final answer.

### Visual CoT (CoVT-Inspired)
Passes the image through SigLIP inside llava-interleave-qwen-0.5b-hf, mean-pools visual tokens, produces pseudo-observations, and constructs a multi-step visual reasoning prompt. Does not use region proposals or training.

### Visual–Text Fusion Reasoner
Iteratively alternates between visual observation extraction and textual reasoning generation. After K steps, all pairs are fused into a final summary prompt used to generate the answer.

## Model and Inference Settings

- **Backbone:** llava-interleave-qwen-0.5b-hf
- **Vision:** SigLIP encoder
- **Language:** Qwen 0.5B
- **Inference:** FP16, greedy decoding, optional 4-bit quantization

No training or fine-tuning is performed.

## Evaluation

Metrics include:
- Accuracy (exact/containment match)
- Average reasoning steps
- Per-question-family accuracy
- Qualitative examples and traces

Outputs:
- `results.json`
- Plots under `data/results/plots/`

## Reproducing Paper Results

```bash
pip install -r requirements.txt
python download_clevr_images.py --clevr-dir /path/to/CLEVR_v1.0 --num-images 200 --split val
python src/run_experiments.py --num-samples 50 --split val --device cuda
python src/generate_paper_artifacts.py
```

## Notes

- Only a small CLEVR image subset is required.
- All methods share the same model instance for fair comparison.
- Image features can be cached for speed.

## Citation

Please cite CLEVR, CoVT, and the associated paper for this project.

## License

For research and educational use only. Check license for details.
