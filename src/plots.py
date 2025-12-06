"""
Plotting and Visualization Module

This module provides functions to visualize evaluation results:
- Accuracy comparison bar charts
- Reasoning steps vs accuracy line charts
- Failure case scatter plots
- Qualitative examples
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont


def plot_accuracy_comparison(
    summary: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Plot bar chart comparing accuracy of all three methods.
    
    Args:
        summary: Summary dictionary from evaluator
        save_path: Optional path to save figure
        figsize: Figure size
    """
    methods = ["Text-only CoT", "CoVT Visual CoT", "Visual + Text Fusion"]
    accuracies = [
        summary["baseline"]["accuracy"],
        summary["covt"]["accuracy"],
        summary["fusion"]["accuracy"]
    ]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    plt.figure(figsize=figsize)
    bars = plt.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Accuracy Comparison: Reasoning Methods', fontsize=14, fontweight='bold')
    plt.ylim(0, max(accuracies) * 1.2)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved accuracy comparison to {save_path}")
    
    plt.show()


def plot_steps_vs_accuracy(
    results: List[Dict],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    Plot line chart showing number of reasoning steps vs accuracy.
    
    Args:
        results: List of detailed results from evaluator
        save_path: Optional path to save figure
        figsize: Figure size
    """
    # Extract data
    baseline_steps = []
    baseline_correct = []
    covt_steps = []
    covt_correct = []
    fusion_steps = []
    fusion_correct = []
    
    for result in results:
        if result.get("ground_truth"):
            baseline_steps.append(result["baseline"]["num_steps"])
            baseline_correct.append(1 if result["baseline"].get("correct") else 0)
            
            covt_steps.append(result["covt"]["num_steps"])
            covt_correct.append(1 if result["covt"].get("correct") else 0)
            
            fusion_steps.append(result["fusion"]["num_steps"])
            fusion_correct.append(1 if result["fusion"].get("correct") else 0)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Baseline
    if baseline_steps:
        axes[0].scatter(baseline_steps, baseline_correct, alpha=0.5, color='#3498db')
        axes[0].set_xlabel('Number of Steps')
        axes[0].set_ylabel('Correct (1) / Incorrect (0)')
        axes[0].set_title('Text-only CoT')
        axes[0].grid(alpha=0.3)
    
    # CoVT
    if covt_steps:
        axes[1].scatter(covt_steps, covt_correct, alpha=0.5, color='#e74c3c')
        axes[1].set_xlabel('Number of Steps')
        axes[1].set_ylabel('Correct (1) / Incorrect (0)')
        axes[1].set_title('CoVT Visual CoT')
        axes[1].grid(alpha=0.3)
    
    # Fusion
    if fusion_steps:
        axes[2].scatter(fusion_steps, fusion_correct, alpha=0.5, color='#2ecc71')
        axes[2].set_xlabel('Number of Steps')
        axes[2].set_ylabel('Correct (1) / Incorrect (0)')
        axes[2].set_title('Visual + Text Fusion')
        axes[2].grid(alpha=0.3)
    
    plt.suptitle('Reasoning Steps vs Accuracy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved steps vs accuracy plot to {save_path}")
    
    plt.show()


def plot_failure_cases(
    results: List[Dict],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    Plot scatter plot showing failure cases across methods.
    
    Args:
        results: List of detailed results from evaluator
        save_path: Optional path to save figure
        figsize: Figure size
    """
    # Count failures for each method
    baseline_failures = sum(1 for r in results 
                           if r.get("ground_truth") and not r["baseline"].get("correct", True))
    covt_failures = sum(1 for r in results 
                       if r.get("ground_truth") and not r["covt"].get("correct", True))
    fusion_failures = sum(1 for r in results 
                         if r.get("ground_truth") and not r["fusion"].get("correct", True))
    
    # Count cases where only one method failed
    only_baseline_failed = 0
    only_covt_failed = 0
    only_fusion_failed = 0
    all_failed = 0
    all_correct = 0
    
    for result in results:
        if not result.get("ground_truth"):
            continue
        
        baseline_correct = result["baseline"].get("correct", False)
        covt_correct = result["covt"].get("correct", False)
        fusion_correct = result["fusion"].get("correct", False)
        
        if not baseline_correct and covt_correct and fusion_correct:
            only_baseline_failed += 1
        elif baseline_correct and not covt_correct and fusion_correct:
            only_covt_failed += 1
        elif baseline_correct and covt_correct and not fusion_correct:
            only_fusion_failed += 1
        elif not baseline_correct and not covt_correct and not fusion_correct:
            all_failed += 1
        elif baseline_correct and covt_correct and fusion_correct:
            all_correct += 1
    
    # Create grouped bar chart
    categories = ['Only Baseline\nFailed', 'Only CoVT\nFailed', 'Only Fusion\nFailed', 
                  'All Failed', 'All Correct']
    counts = [only_baseline_failed, only_covt_failed, only_fusion_failed, 
              all_failed, all_correct]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#95a5a6', '#27ae60']
    
    plt.figure(figsize=figsize)
    bars = plt.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel('Number of Cases', fontsize=12, fontweight='bold')
    plt.title('Failure Case Analysis', fontsize=14, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved failure cases plot to {save_path}")
    
    plt.show()


def visualize_qualitative_example(
    sample: Dict,
    result: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 10)
):
    """
    Visualize a qualitative example with all three methods.
    
    Args:
        sample: Original sample with image and question
        result: Result dictionary from evaluator
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    
    # Original image
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(sample["image"])
    ax1.axis('off')
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    
    # Question
    ax2 = plt.subplot(2, 3, 2)
    ax2.axis('off')
    question_text = f"Question: {sample['question']}\n\n"
    if sample.get("answer"):
        question_text += f"Ground Truth: {sample['answer']}"
    ax2.text(0.1, 0.5, question_text, fontsize=11, 
             verticalalignment='center', wrap=True)
    ax2.set_title('Question & Answer', fontsize=12, fontweight='bold')
    
    # Baseline result
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')
    baseline_text = "Text-only CoT:\n\n"
    baseline_text += f"Answer: {result['baseline']['answer']}\n"
    baseline_text += f"Correct: {result['baseline'].get('correct', 'N/A')}\n\n"
    baseline_text += "Steps:\n"
    for i, step in enumerate(result['baseline']['steps'][:3], 1):
        baseline_text += f"{i}. {step[:80]}...\n"
    ax3.text(0.05, 0.95, baseline_text, fontsize=9,
             verticalalignment='top', wrap=True,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax3.set_title('Baseline', fontsize=12, fontweight='bold')
    
    # CoVT result
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    covt_text = "CoVT Visual CoT:\n\n"
    covt_text += f"Answer: {result['covt']['answer']}\n"
    covt_text += f"Correct: {result['covt'].get('correct', 'N/A')}\n\n"
    covt_text += "Visual Steps:\n"
    for i, step in enumerate(result['covt']['steps'][:3], 1):
        covt_text += f"{i}. {step[:80]}...\n"
    ax4.text(0.05, 0.95, covt_text, fontsize=9,
             verticalalignment='top', wrap=True,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    ax4.set_title('CoVT', fontsize=12, fontweight='bold')
    
    # Fusion result
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    fusion_text = "Visual + Text Fusion:\n\n"
    fusion_text += f"Answer: {result['fusion']['answer']}\n"
    fusion_text += f"Correct: {result['fusion'].get('correct', 'N/A')}\n\n"
    fusion_text += "Fused Steps:\n"
    for i, step in enumerate(result['fusion'].get('fused_steps', [])[:2], 1):
        fusion_text += f"{i}. Visual: {step.get('visual_observation', '')[:40]}...\n"
        fusion_text += f"   Text: {step.get('textual_reasoning', '')[:40]}...\n"
    ax5.text(0.05, 0.95, fusion_text, fontsize=9,
             verticalalignment='top', wrap=True,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax5.set_title('Fusion', fontsize=12, fontweight='bold')
    
    # Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = "Summary:\n\n"
    summary_text += f"Baseline Accuracy: {result['baseline'].get('correct', 'N/A')}\n"
    summary_text += f"CoVT Accuracy: {result['covt'].get('correct', 'N/A')}\n"
    summary_text += f"Fusion Accuracy: {result['fusion'].get('correct', 'N/A')}\n\n"
    summary_text += f"Baseline Steps: {result['baseline']['num_steps']}\n"
    summary_text += f"CoVT Steps: {result['covt']['num_steps']}\n"
    summary_text += f"Fusion Steps: {result['fusion']['num_steps']}"
    ax6.text(0.1, 0.5, summary_text, fontsize=11,
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.set_title('Summary', fontsize=12, fontweight='bold')
    
    plt.suptitle('Qualitative Example Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved qualitative example to {save_path}")
    
    plt.show()


def plot_clevr_question_type_accuracy(
    results: List[Dict],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8)
):
    """
    Plot accuracy by CLEVR question family/type.
    
    Args:
        results: List of detailed results with question_family_index
        save_path: Optional path to save figure
        figsize: Figure size
    """
    # Group by question family if available
    from collections import defaultdict
    
    family_results = defaultdict(lambda: {
        "baseline": {"correct": 0, "total": 0},
        "covt": {"correct": 0, "total": 0},
        "fusion": {"correct": 0, "total": 0}
    })
    
    for result in results:
        family_idx = result.get("question_family_index", "unknown")
        
        if result.get("ground_truth"):
            family_results[family_idx]["baseline"]["total"] += 1
            family_results[family_idx]["covt"]["total"] += 1
            family_results[family_idx]["fusion"]["total"] += 1
            
            if result["baseline"].get("correct"):
                family_results[family_idx]["baseline"]["correct"] += 1
            if result["covt"].get("correct"):
                family_results[family_idx]["covt"]["correct"] += 1
            if result["fusion"].get("correct"):
                family_results[family_idx]["fusion"]["correct"] += 1
    
    # Calculate accuracies
    families = sorted([f for f in family_results.keys() if f != "unknown"])
    baseline_accs = [
        family_results[f]["baseline"]["correct"] / max(family_results[f]["baseline"]["total"], 1)
        for f in families
    ]
    covt_accs = [
        family_results[f]["covt"]["correct"] / max(family_results[f]["covt"]["total"], 1)
        for f in families
    ]
    fusion_accs = [
        family_results[f]["fusion"]["correct"] / max(family_results[f]["fusion"]["total"], 1)
        for f in families
    ]
    
    if not families:
        print("No question family information available")
        return
    
    # Create plot
    x = np.arange(len(families))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - width, baseline_accs, width, label='Text-only CoT', color='#3498db', alpha=0.7)
    bars2 = ax.bar(x, covt_accs, width, label='Visual CoT (CoVT)', color='#e74c3c', alpha=0.7)
    bars3 = ax.bar(x + width, fusion_accs, width, label='Visual+Text Fusion', color='#2ecc71', alpha=0.7)
    
    ax.set_xlabel('Question Family Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy by CLEVR Question Family', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Family {f}" for f in families])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved question type accuracy plot to {save_path}")
    
    plt.show()


def plot_reasoning_length_comparison(
    results: List[Dict],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    Plot comparison of reasoning step lengths across methods.
    
    Args:
        results: List of detailed results
        save_path: Optional path to save figure
        figsize: Figure size
    """
    baseline_lengths = [r["baseline"]["num_steps"] for r in results]
    covt_lengths = [r["covt"]["num_steps"] for r in results]
    fusion_lengths = [r["fusion"]["num_steps"] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].hist(baseline_lengths, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].set_title('Text-only CoT', fontweight='bold')
    axes[0].set_xlabel('Number of Steps')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(covt_lengths, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[1].set_title('Visual CoT (CoVT)', fontweight='bold')
    axes[1].set_xlabel('Number of Steps')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(alpha=0.3)
    
    axes[2].hist(fusion_lengths, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[2].set_title('Visual+Text Fusion', fontweight='bold')
    axes[2].set_xlabel('Number of Steps')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(alpha=0.3)
    
    plt.suptitle('Reasoning Step Length Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved reasoning length comparison to {save_path}")
    
    plt.show()


def plot_all_results(
    summary: Dict,
    results: List[Dict],
    output_dir: str = "./plots"
):
    """
    Generate all plots and save to output directory.
    
    Args:
        summary: Summary dictionary from evaluator
        results: List of detailed results
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Generating plots...")
    
    # Accuracy comparison
    plot_accuracy_comparison(
        summary,
        save_path=f"{output_dir}/accuracy_comparison.png"
    )
    
    # Steps vs accuracy
    plot_steps_vs_accuracy(
        results,
        save_path=f"{output_dir}/steps_vs_accuracy.png"
    )
    
    # Failure cases
    plot_failure_cases(
        results,
        save_path=f"{output_dir}/failure_cases.png"
    )
    
    # CLEVR-specific plots
    try:
        plot_clevr_question_type_accuracy(
            results,
            save_path=f"{output_dir}/question_type_accuracy.png"
        )
    except Exception as e:
        print(f"Could not generate question type plot: {e}")
    
    plot_reasoning_length_comparison(
        results,
        save_path=f"{output_dir}/reasoning_length_comparison.png"
    )
    
    print(f"All plots saved to {output_dir}")
