#!/usr/bin/env python3
"""
Experiment Runner for Interleave-Qwen Reasoning

This version is rewritten to be fully compatible with:
    llava-interleave-qwen-0.5b-hf

It NO LONGER loads separate models internally.
Instead, the notebook injects:
    model, tokenizer, image_processor
as global objects before calling run_experiments().
"""

import sys
import os
from pathlib import Path
import json
import argparse
from typing import Dict, List
from tqdm import tqdm

# NOTE: Reasoners are provided by notebook, not imported from src
# We expect the notebook to define:
#   baseline_reasoner
#   covt_reasoner
#   fusion_reasoner

def compute_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    correct = 0
    total = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        pred = str(pred).lower().strip()
        gt = str(gt).lower().strip()

        import string
        pred = pred.translate(str.maketrans('', '', string.punctuation))
        gt = gt.translate(str.maketrans('', '', string.punctuation))

        if pred == gt or gt in pred or pred in gt:
            correct += 1

    return correct / total if total > 0 else 0.0



def run_experiments(
    questions_path: str,
    images_dir: str,
    split: str,
    num_samples: int,
    device: str,
    load_in_4bit: bool,
    output_dir: str
) -> Dict:

    # ------------------------------------------
    # VERIFY NOTEBOOK HAS INJECTED THE REASONERS
    # ------------------------------------------
    global baseline_reasoner, covt_reasoner, fusion_reasoner

    missing = []
    if "baseline_reasoner" not in globals():
        missing.append("baseline_reasoner")
    if "covt_reasoner" not in globals():
        missing.append("covt_reasoner")
    if "fusion_reasoner" not in globals():
        missing.append("fusion_reasoner")

    if missing:
        raise RuntimeError(f"❌ Missing reasoners injected from notebook: {missing}")

    # ------------------------------------------
    # Load CLEVR dataset
    # ------------------------------------------
    from src.dataset_clevr import load_clevr_dataset

    print("=" * 60)
    print("CLEVR Reasoning Experiments (Qwen-Interleave Compatible)")
    print("=" * 60)
    print(f"Samples: {num_samples}")
    print(f"Split: {split}")
    print(f"Device: {device}")
    print("=" * 60)

    samples = load_clevr_dataset(
        questions_path=questions_path,
        images_dir=images_dir,
        split=split,
        num_samples=num_samples
    )

    if len(samples) == 0:
        raise RuntimeError("❌ No CLEVR samples loaded!")

    print(f"✓ Loaded {len(samples)} samples.\n")

    # ------------------------------------------
    # Storage
    # ------------------------------------------
    results = {
        "baseline": {"pred": [], "gt": [], "steps": [], "ok": []},
        "covt": {"pred": [], "gt": [], "steps": [], "ok": []},
        "fusion": {"pred": [], "gt": [], "steps": [], "ok": []},
    }

    # ------------------------------------------
    # Main evaluation loop
    # ------------------------------------------
    print("=" * 60)
    print("Running evaluation...")
    print("=" * 60)

    for idx, sample in enumerate(tqdm(samples, desc="Samples")):
        img = sample["image"]
        q = sample["question"]
        gt = sample.get("answer", "").strip()

        # 1) Baseline
        try:
            b = baseline_reasoner.reason(img, q)
            pred = b.get("answer", "")
            results["baseline"]["pred"].append(pred)
            results["baseline"]["gt"].append(gt)
            results["baseline"]["steps"].append(b.get("steps", []))
            results["baseline"]["ok"].append(pred.lower().strip() == gt.lower().strip())
        except Exception as e:
            results["baseline"]["pred"].append("")
            results["baseline"]["gt"].append(gt)
            results["baseline"]["steps"].append([])
            results["baseline"]["ok"].append(False)

        # 2) Visual CoT (may fail → fallback)
        try:
            c = covt_reasoner.reason(img, q)
            pred = c.get("answer", "")
            results["covt"]["pred"].append(pred)
            results["covt"]["gt"].append(gt)
            results["covt"]["steps"].append(c.get("steps", []))
            results["covt"]["ok"].append(pred.lower().strip() == gt.lower().strip())
        except Exception:
            # Graceful fallback
            results["covt"]["pred"].append("")
            results["covt"]["gt"].append(gt)
            results["covt"]["steps"].append([])
            results["covt"]["ok"].append(False)

        # 3) Fusion Reasoner
        try:
            f = fusion_reasoner.reason(img, q)
            pred = f.get("answer", "")
            results["fusion"]["pred"].append(pred)
            results["fusion"]["gt"].append(gt)
            results["fusion"]["steps"].append(f.get("fused_steps", []))
            results["fusion"]["ok"].append(pred.lower().strip() == gt.lower().strip())
        except Exception:
            results["fusion"]["pred"].append("")
            results["fusion"]["gt"].append(gt)
            results["fusion"]["steps"].append([])
            results["fusion"]["ok"].append(False)

    # ------------------------------------------
    # Compute accuracies
    # ------------------------------------------
    baseline_acc = compute_accuracy(results["baseline"]["pred"], results["baseline"]["gt"])
    covt_acc = compute_accuracy(results["covt"]["pred"], results["covt"]["gt"])
    fusion_acc = compute_accuracy(results["fusion"]["pred"], results["fusion"]["gt"])

    summary = {
        "text_only_cot_accuracy": baseline_acc,
        "visual_cot_accuracy": covt_acc,
        "fusion_accuracy": fusion_acc,
    }

    # ------------------------------------------
    # Save summary
    # ------------------------------------------
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"

    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline (Text-only CoT): {baseline_acc:.2%}")
    print(f"Visual CoT (CoVT):       {covt_acc:.2%}")
    print(f"Fusion Reasoner:         {fusion_acc:.2%}")
    print("=" * 60)
    print(f"✓ Saved results to: {out_path}")

    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=str, required=True)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--4bit", action="store_true")

    args = parser.parse_args()

    print("❌ You cannot run this script directly.")
    print("This script is invoked from inside the notebook where models + reasoners are loaded.")
