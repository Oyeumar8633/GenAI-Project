#!/usr/bin/env python3
"""
CLEVR Dataset Path Verification Script

Verifies that all required CLEVR dataset files and paths are correctly set up.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Set


def verify_clevr_setup(
    questions_path: str = "data/clevr_questions.json",
    images_dir: str = "data/images_subset"
) -> Dict:
    """
    Verify CLEVR dataset setup.
    
    Args:
        questions_path: Path to CLEVR questions JSON file
        images_dir: Directory containing CLEVR images
        
    Returns:
        Dictionary with verification results
    """
    results = {
        "questions_file_exists": False,
        "images_dir_exists": False,
        "num_images": 0,
        "num_questions": 0,
        "matching_samples": 0,
        "missing_images": [],
        "warnings": [],
        "errors": []
    }
    
    print("=" * 60)
    print("CLEVR Dataset Path Verification")
    print("=" * 60)
    
    # Check questions file
    questions_path_obj = Path(questions_path)
    if questions_path_obj.exists():
        results["questions_file_exists"] = True
        print(f"✓ Questions file found: {questions_path}")
        
        try:
            with open(questions_path, 'r') as f:
                questions_data = json.load(f)
                questions = questions_data.get('questions', [])
                results["num_questions"] = len(questions)
                print(f"✓ Loaded {len(questions)} questions")
        except Exception as e:
            results["errors"].append(f"Failed to load questions JSON: {e}")
            print(f"✗ Error loading questions: {e}")
    else:
        results["errors"].append(f"Questions file not found: {questions_path}")
        print(f"✗ Questions file not found: {questions_path}")
        return results
    
    # Check images directory
    images_dir_obj = Path(images_dir)
    if images_dir_obj.exists() and images_dir_obj.is_dir():
        results["images_dir_exists"] = True
        print(f"✓ Images directory found: {images_dir}")
        
        # Count images
        image_files = list(images_dir_obj.glob("CLEVR_*.png"))
        results["num_images"] = len(image_files)
        print(f"✓ Found {len(image_files)} images")
        
        if len(image_files) == 0:
            results["warnings"].append("No CLEVR images found in images directory")
            print("⚠ Warning: No CLEVR images found")
    else:
        results["errors"].append(f"Images directory not found: {images_dir}")
        print(f"✗ Images directory not found: {images_dir}")
        return results
    
    # Check for matching images and questions
    if results["questions_file_exists"] and results["images_dir_exists"]:
        print(f"\nChecking image-question matches...")
        
        # Get set of image filenames
        image_filenames = {img.name for img in image_files}
        
        # Check which questions have matching images
        matching_count = 0
        missing_images = []
        
        for q in questions:
            img_filename = q.get('image_filename', '')
            if img_filename in image_filenames:
                matching_count += 1
            else:
                missing_images.append(img_filename)
        
        results["matching_samples"] = matching_count
        results["missing_images"] = missing_images[:10]  # Store first 10 for display
        
        print(f"✓ Found {matching_count} questions with matching images")
        
        if missing_images:
            missing_count = len(missing_images)
            print(f"⚠ Warning: {missing_count} questions reference images not in images_subset/")
            if missing_count <= 10:
                print("  Missing images:")
                for img in missing_images:
                    print(f"    - {img}")
            else:
                print(f"  Missing images (showing first 10):")
                for img in missing_images[:10]:
                    print(f"    - {img}")
                print(f"    ... and {missing_count - 10} more")
            
            results["warnings"].append(f"{missing_count} questions reference missing images")
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    if results["errors"]:
        print("✗ ERRORS FOUND:")
        for error in results["errors"]:
            print(f"  - {error}")
        print("\nPlease fix these errors before proceeding.")
    else:
        print("✓ No critical errors found")
    
    if results["warnings"]:
        print("\n⚠ WARNINGS:")
        for warning in results["warnings"]:
            print(f"  - {warning}")
    
    print(f"\nDataset Statistics:")
    print(f"  Questions: {results['num_questions']}")
    print(f"  Images: {results['num_images']}")
    print(f"  Matching samples: {results['matching_samples']}")
    
    if results["matching_samples"] > 0:
        print(f"\n✓ Ready to use! {results['matching_samples']} samples available for reasoning.")
    else:
        print(f"\n✗ No matching samples found. Please check your dataset setup.")
    
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify CLEVR dataset paths")
    parser.add_argument(
        "--questions",
        type=str,
        default="data/clevr_questions.json",
        help="Path to CLEVR questions JSON file"
    )
    parser.add_argument(
        "--images",
        type=str,
        default="data/images_subset",
        help="Path to CLEVR images directory"
    )
    
    args = parser.parse_args()
    
    verify_clevr_setup(args.questions, args.images)
