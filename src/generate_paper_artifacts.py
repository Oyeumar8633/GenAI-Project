#!/usr/bin/env python3
"""
Generate Paper Artifacts for Research Paper

Creates visualizations, examples, and metrics for the research paper.
"""

import sys
import json
import csv
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path.parent))

from src.dataset_clevr import load_clevr_dataset
from src.reasoning_baseline import TextOnlyCoTReasoner
from src.reasoning_cvt import CoVTVisualReasoner
from src.reasoning_fusion import VisualTextFusionReasoner


def save_reasoning_examples(
    samples: List[Dict],
    results: Dict,
    output_dir: Path,
    num_examples: int = 5
):
    """
    Save reasoning examples for each method.
    
    Args:
        samples: Dataset samples
        results: Results dictionary
        output_dir: Output directory
        num_examples: Number of examples to save
    """
    examples_dir = output_dir / "reasoning_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    # Select examples (mix of correct and incorrect)
    selected_indices = []
    
    # Get some correct examples
    for idx, result in enumerate(results.get("samples", [])[:num_examples * 2]):
        if result.get("baseline", {}).get("correct") or \
           result.get("covt", {}).get("correct") or \
           result.get("fusion", {}).get("correct"):
            selected_indices.append(idx)
            if len(selected_indices) >= num_examples:
                break
    
    # Fill remaining with any examples
    while len(selected_indices) < num_examples and len(selected_indices) < len(samples):
        if len(selected_indices) not in selected_indices:
            selected_indices.append(len(selected_indices))
    
    selected_indices = selected_indices[:num_examples]
    
    examples_md = []
    examples_md.append("# Reasoning Examples\n\n")
    examples_md.append(f"This document shows {num_examples} examples of reasoning from each method.\n\n")
    
    for ex_idx, sample_idx in enumerate(selected_indices, 1):
        if sample_idx >= len(samples):
            continue
            
        sample = samples[sample_idx]
        result = results.get("samples", [])[sample_idx] if sample_idx < len(results.get("samples", [])) else {}
        
        examples_md.append(f"## Example {ex_idx}\n\n")
        examples_md.append(f"**Question:** {sample['question']}\n\n")
        examples_md.append(f"**Ground Truth:** {sample.get('answer', 'N/A')}\n\n")
        examples_md.append(f"**Image:** {sample.get('image_filename', 'N/A')}\n\n")
        
        # Baseline
        examples_md.append("### Text-only CoT (Baseline)\n\n")
        baseline_result = result.get("baseline", {})
        examples_md.append(f"**Prediction:** {baseline_result.get('prediction', 'N/A')}\n\n")
        examples_md.append(f"**Correct:** {baseline_result.get('correct', False)}\n\n")
        examples_md.append(f"**Steps:** {baseline_result.get('num_steps', 0)}\n\n")
        
        # CoVT
        examples_md.append("### Visual CoT (CoVT-style)\n\n")
        covt_result = result.get("covt", {})
        examples_md.append(f"**Prediction:** {covt_result.get('prediction', 'N/A')}\n\n")
        examples_md.append(f"**Correct:** {covt_result.get('correct', False)}\n\n")
        examples_md.append(f"**Steps:** {covt_result.get('num_steps', 0)}\n\n")
        
        # Fusion
        examples_md.append("### Visual + Text Fusion (Novel)\n\n")
        fusion_result = result.get("fusion", {})
        examples_md.append(f"**Prediction:** {fusion_result.get('prediction', 'N/A')}\n\n")
        examples_md.append(f"**Correct:** {fusion_result.get('correct', False)}\n\n")
        examples_md.append(f"**Steps:** {fusion_result.get('num_steps', 0)}\n\n")
        
        examples_md.append("---\n\n")
    
    # Save markdown
    with open(examples_dir / "reasoning_examples.md", 'w') as f:
        f.write("".join(examples_md))
    
    print(f"✓ Saved reasoning examples to {examples_dir / 'reasoning_examples.md'}")


def generate_confusion_matrix(
    results: Dict,
    output_dir: Path
):
    """
    Generate confusion matrix CSV for analysis.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
    """
    samples = results.get("samples", [])
    
    # Collect predictions and ground truths
    data = []
    for sample in samples:
        data.append({
            "question": sample.get("question", ""),
            "ground_truth": sample.get("ground_truth", ""),
            "baseline_prediction": sample.get("baseline", {}).get("prediction", ""),
            "covt_prediction": sample.get("covt", {}).get("prediction", ""),
            "fusion_prediction": sample.get("fusion", {}).get("prediction", ""),
            "baseline_correct": sample.get("baseline", {}).get("correct", False),
            "covt_correct": sample.get("covt", {}).get("correct", False),
            "fusion_correct": sample.get("fusion", {}).get("correct", False)
        })
    
    # Save as CSV
    csv_path = output_dir / "confusion_matrix.csv"
    with open(csv_path, 'w', newline='') as f:
        if data:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    
    print(f"✓ Saved confusion matrix to {csv_path}")


def generate_metrics_summary(
    results: Dict,
    output_dir: Path
):
    """
    Generate metrics summary JSON.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
    """
    summary = results.get("summary", {})
    
    metrics = {
        "baseline": {
            "accuracy": summary.get("baseline", {}).get("accuracy", 0),
            "avg_steps": summary.get("baseline", {}).get("avg_steps", 0),
            "correct": summary.get("baseline", {}).get("correct", 0),
            "total": summary.get("baseline", {}).get("total", 0)
        },
        "covt": {
            "accuracy": summary.get("covt", {}).get("accuracy", 0),
            "avg_steps": summary.get("covt", {}).get("avg_steps", 0),
            "correct": summary.get("covt", {}).get("correct", 0),
            "total": summary.get("covt", {}).get("total", 0)
        },
        "fusion": {
            "accuracy": summary.get("fusion", {}).get("accuracy", 0),
            "avg_steps": summary.get("fusion", {}).get("avg_steps", 0),
            "correct": summary.get("fusion", {}).get("correct", 0),
            "total": summary.get("fusion", {}).get("total", 0)
        },
        "metadata": results.get("metadata", {})
    }
    
    json_path = output_dir / "metrics_summary.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Saved metrics summary to {json_path}")


def generate_paper_artifacts(
    results_file: str = "data/results/results.json",
    output_dir: str = "paper_artifacts",
    num_examples: int = 5
):
    """
    Generate all paper artifacts.
    
    Args:
        results_file: Path to results JSON file
        output_dir: Output directory for artifacts
        num_examples: Number of examples to generate
    """
    print("=" * 60)
    print("Generating Paper Artifacts")
    print("=" * 60)
    
    # Load results
    results_path = Path(results_file)
    if not results_path.exists():
        print(f"✗ Error: Results file not found: {results_path}")
        print("Please run experiments first: python src/run_experiments.py")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_path}")
    
    # Load samples for examples (if needed)
    samples = []
    try:
        samples = load_clevr_dataset(
            questions_path="data/clevr_questions.json",
            images_dir="data/images_subset",
            split="val",
            num_samples=100
        )
    except Exception as e:
        print(f"Warning: Could not load samples for examples: {e}")
    
    # Generate artifacts
    print("\n1. Generating reasoning examples...")
    save_reasoning_examples(samples, results, output_path, num_examples)
    
    print("\n2. Generating confusion matrix...")
    generate_confusion_matrix(results, output_path)
    
    print("\n3. Generating metrics summary...")
    generate_metrics_summary(results, output_path)
    
    print("\n" + "=" * 60)
    print("✓ All paper artifacts generated!")
    print(f"✓ Saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate paper artifacts")
    parser.add_argument(
        "--results",
        type=str,
        default="data/results/results.json",
        help="Path to results JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper_artifacts",
        help="Output directory for artifacts"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of examples to generate"
    )
    
    args = parser.parse_args()
    
    generate_paper_artifacts(
        results_file=args.results,
        output_dir=args.output,
        num_examples=args.num_examples
    )
