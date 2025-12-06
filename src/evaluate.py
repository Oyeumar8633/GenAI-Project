"""
Evaluation Module for Comparing Reasoning Methods

This module provides functions to evaluate and compare:
1. CoVT-style Visual CoT
2. Visual + Text Fusion (Novel)
3. Text-only CoT (Baseline)
"""

import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm
from .reasoning_baseline import TextOnlyCoTReasoner
from .reasoning_cvt import CoVTVisualReasoner
from .reasoning_fusion import VisualTextFusionReasoner


class ReasoningEvaluator:
    """
    Evaluator for comparing different reasoning methods.
    """
    
    def __init__(
        self,
        baseline_reasoner: TextOnlyCoTReasoner,
        covt_reasoner: CoVTVisualReasoner,
        fusion_reasoner: VisualTextFusionReasoner
    ):
        """
        Initialize evaluator with three reasoners.
        
        Args:
            baseline_reasoner: Text-only CoT reasoner
            covt_reasoner: CoVT-style visual reasoner
            fusion_reasoner: Visual-text fusion reasoner
        """
        self.baseline_reasoner = baseline_reasoner
        self.covt_reasoner = covt_reasoner
        self.fusion_reasoner = fusion_reasoner
    
    def evaluate_sample(
        self,
        image,
        question: str,
        ground_truth: Optional[str] = None
    ) -> Dict:
        """
        Evaluate a single sample with all three methods.
        
        Args:
            image: PIL Image
            question: Question text
            ground_truth: Ground truth answer (optional)
            
        Returns:
            Dictionary with results from all methods
        """
        results = {
            "question": question,
            "ground_truth": ground_truth
        }
        
        # Run baseline
        print("Running baseline text-only CoT...")
        baseline_result = self.baseline_reasoner.reason(image, question)
        results["baseline"] = {
            "steps": baseline_result["steps"],
            "answer": baseline_result["answer"],
            "num_steps": len(baseline_result["steps"])
        }
        
        # Run CoVT
        print("Running CoVT-style visual CoT...")
        covt_result = self.covt_reasoner.reason(image, question)
        results["covt"] = {
            "steps": covt_result["steps"],
            "answer": covt_result["answer"],
            "num_steps": len(covt_result["steps"]),
            "has_visual_evidence": True
        }
        
        # Run fusion
        print("Running visual + text fusion...")
        fusion_result = self.fusion_reasoner.reason(image, question)
        results["fusion"] = {
            "visual_steps": fusion_result["visual_steps"],
            "text_steps": fusion_result["text_steps"],
            "fused_steps": fusion_result["fused_steps"],
            "answer": fusion_result["answer"],
            "num_steps": len(fusion_result["fused_steps"]),
            "has_visual_evidence": True
        }
        
        # Compute accuracy if ground truth available
        if ground_truth:
            results["baseline"]["correct"] = self._check_answer(
                baseline_result["answer"], ground_truth
            )
            results["covt"]["correct"] = self._check_answer(
                covt_result["answer"], ground_truth
            )
            results["fusion"]["correct"] = self._check_answer(
                fusion_result["answer"], ground_truth
            )
        
        return results
    
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if predicted answer matches ground truth.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            True if answers match (case-insensitive, normalized)
        """
        # Normalize answers
        pred_norm = predicted.lower().strip()
        gt_norm = ground_truth.lower().strip()
        
        # Remove punctuation
        import string
        pred_norm = pred_norm.translate(str.maketrans('', '', string.punctuation))
        gt_norm = gt_norm.translate(str.maketrans('', '', string.punctuation))
        
        # Check exact match
        if pred_norm == gt_norm:
            return True
        
        # Check if ground truth is contained in prediction
        if gt_norm in pred_norm or pred_norm in gt_norm:
            return True
        
        return False
    
    def evaluate_dataset(
        self,
        dataset: List[Dict],
        max_samples: Optional[int] = None,
        save_results: Optional[str] = None
    ) -> Dict:
        """
        Evaluate all three methods on a dataset.
        
        Args:
            dataset: List of samples, each with 'image', 'question', 'answer'
            max_samples: Maximum number of samples to evaluate
            save_results: Optional path to save results JSON
            
        Returns:
            Dictionary with aggregated results
        """
        if max_samples:
            dataset = dataset[:max_samples]
        
        all_results = []
        baseline_correct = 0
        covt_correct = 0
        fusion_correct = 0
        
        baseline_steps = []
        covt_steps = []
        fusion_steps = []
        
        print(f"Evaluating {len(dataset)} samples...")
        
        for idx, sample in enumerate(tqdm(dataset)):
            try:
                result = self.evaluate_sample(
                    image=sample["image"],
                    question=sample["question"],
                    ground_truth=sample.get("answer")
                )
                
                all_results.append(result)
                
                # Aggregate metrics
                if sample.get("answer"):
                    if result["baseline"].get("correct"):
                        baseline_correct += 1
                    if result["covt"].get("correct"):
                        covt_correct += 1
                    if result["fusion"].get("correct"):
                        fusion_correct += 1
                
                baseline_steps.append(result["baseline"]["num_steps"])
                covt_steps.append(result["covt"]["num_steps"])
                fusion_steps.append(result["fusion"]["num_steps"])
                
            except Exception as e:
                print(f"Error evaluating sample {idx}: {e}")
                continue
        
        # Compute final metrics
        total = len(all_results)
        num_with_answers = sum(1 for r in all_results if r.get("ground_truth"))
        
        summary = {
            "total_samples": total,
            "samples_with_answers": num_with_answers,
            "baseline": {
                "accuracy": baseline_correct / num_with_answers if num_with_answers > 0 else 0,
                "correct": baseline_correct,
                "avg_steps": np.mean(baseline_steps) if baseline_steps else 0,
                "has_visual_evidence": False
            },
            "covt": {
                "accuracy": covt_correct / num_with_answers if num_with_answers > 0 else 0,
                "correct": covt_correct,
                "avg_steps": np.mean(covt_steps) if covt_steps else 0,
                "has_visual_evidence": True
            },
            "fusion": {
                "accuracy": fusion_correct / num_with_answers if num_with_answers > 0 else 0,
                "correct": fusion_correct,
                "avg_steps": np.mean(fusion_steps) if fusion_steps else 0,
                "has_visual_evidence": True
            },
            "detailed_results": all_results
        }
        
        # Save results if requested
        if save_results:
            # Convert PIL Images to paths for JSON serialization
            serializable_results = []
            for result in all_results:
                serializable_result = {
                    "question": result["question"],
                    "ground_truth": result.get("ground_truth"),
                    "baseline": {
                        "steps": result["baseline"]["steps"],
                        "answer": result["baseline"]["answer"],
                        "num_steps": result["baseline"]["num_steps"],
                        "correct": result["baseline"].get("correct")
                    },
                    "covt": {
                        "steps": result["covt"]["steps"],
                        "answer": result["covt"]["answer"],
                        "num_steps": result["covt"]["num_steps"],
                        "correct": result["covt"].get("correct")
                    },
                    "fusion": {
                        "visual_steps": result["fusion"]["visual_steps"],
                        "text_steps": result["fusion"]["text_steps"],
                        "answer": result["fusion"]["answer"],
                        "num_steps": result["fusion"]["num_steps"],
                        "correct": result["fusion"].get("correct")
                    }
                }
                serializable_results.append(serializable_result)
            
            with open(save_results, 'w') as f:
                json.dump({
                    "summary": summary,
                    "results": serializable_results
                }, f, indent=2)
            
            print(f"Results saved to {save_results}")
        
        return summary
    
    def generate_comparison_table(self, summary: Dict) -> str:
        """
        Generate a formatted comparison table.
        
        Args:
            summary: Summary dictionary from evaluate_dataset
            
        Returns:
            Formatted table string
        """
        baseline = summary["baseline"]
        covt = summary["covt"]
        fusion = summary["fusion"]
        
        table = f"""
| Method                     | Accuracy | Avg Reason Steps | Visual Evidence? |
|----------------------------|----------|------------------|------------------|
| CoVT-style Visual CoT      | {covt['accuracy']:.2%}      | {covt['avg_steps']:.1f}               | Yes              |
| Visual + Text Fusion (New) | {fusion['accuracy']:.2%}      | {fusion['avg_steps']:.1f}               | Yes (Better)     |
| Text-only CoT              | {baseline['accuracy']:.2%}      | {baseline['avg_steps']:.1f}               | No               |
"""
        return table
