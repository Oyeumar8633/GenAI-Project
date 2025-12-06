"""
Chain-of-Visual-Thought (CoVT) Reproduction Project

This package implements:
1. Original CoVT-style visual reasoning
2. Novel Visual + Text CoT fusion
3. Baseline text-only CoT reasoning

Uses CLEVR dataset for visual reasoning experiments.
"""

from .dataset_clevr import (
    CLEVRDataset,
    load_clevr_subset,
    load_clevr_dataset,
    get_clevr_split,
    load_clevr_from_questions_file,
    download_clevr_data
)
from .visual_utils import VisualRegionDetector, visualize_regions
from .reasoning_baseline import TextOnlyCoTReasoner, cot_text_only
from .reasoning_cvt import CoVTVisualReasoner, coct_visual_reason
from .reasoning_fusion import VisualTextFusionReasoner, coct_visual_text_fused
from .evaluate import ReasoningEvaluator
from .plots import (
    plot_accuracy_comparison,
    plot_steps_vs_accuracy,
    plot_failure_cases,
    visualize_qualitative_example,
    plot_all_results
)

__version__ = "2.0.0"
__all__ = [
    "CLEVRDataset",
    "load_clevr_subset",
    "load_clevr_dataset",
    "get_clevr_split",
    "load_clevr_from_questions_file",
    "download_clevr_data",
    "VisualRegionDetector",
    "visualize_regions",
    "TextOnlyCoTReasoner",
    "cot_text_only",
    "CoVTVisualReasoner",
    "coct_visual_reason",
    "VisualTextFusionReasoner",
    "coct_visual_text_fused",
    "ReasoningEvaluator",
    "plot_accuracy_comparison",
    "plot_steps_vs_accuracy",
    "plot_failure_cases",
    "visualize_qualitative_example",
    "plot_all_results"
]
