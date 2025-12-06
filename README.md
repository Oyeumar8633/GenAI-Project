# Chain-of-Visual-Thought (CoVT) Reproduction Project

This project reproduces and extends the Chain-of-Visual-Thought (CoVT) approach, implementing a novel visual-text fused reasoning pipeline with comparative experiments on the **CLEVR dataset**.

## Project Overview

This project implements three reasoning methods:

1. **CoVT-style Visual CoT**: Original visual chain-of-thought reasoning using image regions
2. **Visual + Text Fusion (Novel)**: Novel extension that fuses visual observations and textual reasoning at each step
3. **Text-only CoT (Baseline)**: Baseline text-only chain-of-thought from LLaVA

## Dataset: CLEVR

This project uses **CLEVR** (Compositional Language and Elementary Visual Reasoning), a synthetic visual reasoning dataset that is:
- **Small and manageable** (~86MB for questions, small image subset needed)
- **Perfect for visual reasoning** - designed for compositional reasoning tasks
- **Stable and well-structured** - synthetic images with clear ground truth

### CLEVR Dataset Structure

- **Questions JSON**: Contains questions, answers, and image filename mappings
- **Images**: Synthetic scenes with objects (only need 100-300 images for testing)
- **Answers**: Simple strings ("yes", "no", "red", "3", etc.)

## Project Structure

```
/project
    /data
        clevr_questions.json      # CLEVR questions JSON file
        images_subset/            # Small subset of CLEVR images
    /models                       # Model cache directory
    /src
        dataset_clevr.py          # CLEVR dataset loader
        visual_utils.py            # Image processing and region detection
        reasoning_baseline.py      # Text-only CoT reasoning
        reasoning_cvt.py           # Original CoVT-style reasoning
        reasoning_fusion.py       # Novel visual-text fusion
        evaluate.py                # Evaluation and comparison
        plots.py                   # Visualization functions
    main.ipynb                     # Main notebook for end-to-end pipeline
    requirements.txt               # Python dependencies
    README.md                      # This file
```

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Setup CLEVR dataset:

   **If you have CLEVR_v1.0 directory locally:**
   ```bash
   # Extract images from local CLEVR directory
   python download_clevr_images.py --clevr-dir /path/to/CLEVR_v1.0 --num-images 200 --split val
   
   # Copy questions file
   cp /path/to/CLEVR_v1.0/questions/CLEVR_val_questions.json data/clevr_questions.json
   ```
   
   **If you need to download CLEVR:**
   - Download CLEVR v1.0 (no images) - 86 MB:
     - https://cs.stanford.edu/people/jcjohns/clevr/CLEVR_v1.0_no_images.zip
     - Extract and place `CLEVR_v1.0/questions/CLEVR_val_questions.json` as `data/clevr_questions.json`
   
   - Download CLEVR Images:
     - Train: https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_train.zip (13GB)
     - Val: https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_val.zip (5GB)
     - Extract and use `--clevr-dir` option with the script

## Usage

### Quick Start - Run Experiments

**Option 1: Using the Experiment Runner (Recommended)**

```bash
# 1. Setup CLEVR dataset (if you have local CLEVR_v1.0 directory)
python download_clevr_images.py --clevr-dir /path/to/CLEVR_v1.0 --num-images 200 --split val

# 2. Verify dataset setup
python src/verify_clevr_paths.py

# 3. Run experiments
python src/run_experiments.py --num-samples 50 --split val --device cuda

# 4. Generate paper artifacts
python src/generate_paper_artifacts.py
```

**Option 2: Using Jupyter Notebook**

Open `main.ipynb` in Jupyter Notebook or Google Colab and run all cells. The notebook will:

1. Set up CLEVR dataset from local directory
2. Verify dataset setup
3. Display CLEVR samples
4. Initialize all three reasoning models
5. Test individual methods
6. Run full experiments
7. Generate visualizations
8. Create paper artifacts

### Programmatic Usage

```python
from src import (
    load_clevr_from_questions_file,
    TextOnlyCoTReasoner,
    CoVTVisualReasoner,
    VisualTextFusionReasoner,
    ReasoningEvaluator
)

# Load CLEVR dataset
samples = load_clevr_from_questions_file(
    questions_path="data/clevr_questions.json",
    images_dir="data/images_subset",
    split="val",
    num_samples=100
)

# Initialize reasoners
baseline = TextOnlyCoTReasoner(device="cuda")
covt = CoVTVisualReasoner(device="cuda", num_visual_steps=3)
fusion = VisualTextFusionReasoner(device="cuda", num_fusion_steps=4)

# Initialize evaluator
evaluator = ReasoningEvaluator(baseline, covt, fusion)

# Evaluate
summary = evaluator.evaluate_dataset(
    samples,
    max_samples=50,
    save_results="results.json"
)

# Print comparison table
print(evaluator.generate_comparison_table(summary))
```

### Individual Reasoning Methods

#### Text-only CoT (Baseline)
```python
from src import cot_text_only
from PIL import Image

result = cot_text_only(image, question)
print(result["answer"])
print(result["steps"])
```

#### CoVT-style Visual CoT
```python
from src import coct_visual_reason

result = coct_visual_reason(image, question, num_steps=3)
print(result["answer"])
print(result["steps"])
```

#### Visual + Text Fusion
```python
from src import coct_visual_text_fused

result = coct_visual_text_fused(image, question, num_steps=4)
print(result["answer"])
print(result["visual_steps"])
print(result["text_steps"])
```

## Models

The project uses:
- **Primary**: LLaVA 1.5 (7B) from HuggingFace: `llava-hf/llava-1.5-7b-hf`
- **Optional**: BLIP-2 for comparison: `Salesforce/blip2-flan-t5-xl`

Models are loaded with:
- bfloat16 precision (default)
- 4-bit quantization (optional, for memory efficiency)
- 8-bit quantization (optional)

## Evaluation Metrics

The evaluation produces:
1. **Accuracy**: Percentage of correct answers (CLEVR has simple string answers)
2. **Average Reasoning Steps**: Mean number of reasoning steps per method
3. **Visual Evidence**: Whether method uses visual regions
4. **Qualitative Examples**: Side-by-side comparisons

## Outputs

The evaluation generates:
- Comparison table (accuracy, steps, visual evidence)
- Bar chart: Accuracy comparison
- Line chart: Reasoning steps vs accuracy
- Scatter plot: Failure case analysis
- Qualitative examples with visualizations

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- ~15GB disk space for models
- Small CLEVR image subset (100-300 images, ~50-150MB)

## Running Experiments

### Full Experiment Pipeline

```bash
# 1. Setup dataset
python download_clevr_images.py --clevr-dir /path/to/CLEVR_v1.0 --num-images 200 --split val

# 2. Verify setup
python src/verify_clevr_paths.py

# 3. Run experiments (evaluates all three methods)
python src/run_experiments.py --num-samples 50 --split val --device cuda

# 4. Generate visualizations (in Python)
python -c "
from src.plots import plot_all_results
import json
with open('data/results/results.json') as f:
    data = json.load(f)
plot_all_results(data['summary'], data['samples'], 'data/results/plots')
"

# 5. Generate paper artifacts
python src/generate_paper_artifacts.py
```

### Expected Output

After running experiments, you'll get:

- **Results**: `data/results/results.json` - Full results with predictions
- **Plots**: `data/results/plots/` - Accuracy comparisons, visualizations
- **Artifacts**: `paper_artifacts/` - Examples, metrics, confusion matrix

### Results Table Format

```
Method                  Accuracy    Avg Steps
--------------------------------------------
Text-only CoT           XX%        X.X
Visual CoT (CoVT)       XX%        X.X
Visual+Text Fusion      XX%        X.X
```

## Reproducing Results

To reproduce the experiments:

1. **Setup environment:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup CLEVR dataset:**
   ```bash
   python download_clevr_images.py --clevr-dir /path/to/CLEVR_v1.0 --num-images 200 --split val
   ```

3. **Run experiments:**
   ```bash
   python src/run_experiments.py --num-samples 50 --split val --device cuda
   ```

4. **View results:**
   ```bash
   cat data/results/results.json | python -m json.tool
   ```

## Notes

- The dataset loader supports quick testing with small subsets (100-300 samples)
- Image embeddings are cached for speed optimization
- All functions are thoroughly documented
- The pipeline is designed to run end-to-end in Jupyter notebooks
- CLEVR answers are simple strings, making evaluation straightforward
- Experiments can be run via script or notebook - both produce same results

## Citation

If you use this code, please cite the original CoVT paper and acknowledge this implementation.

## License

This project is for research purposes. Please check licenses for:
- CLEVR dataset
- LLaVA model
- CLIP model
