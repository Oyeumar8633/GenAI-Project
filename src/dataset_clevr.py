"""
CLEVR Dataset Loader

This module provides functionality to load and prepare the CLEVR dataset
for visual reasoning experiments.

CLEVR is a synthetic visual reasoning dataset with:
- Simple questions about scenes with objects
- Answers are short strings ("yes", "no", "red", "3", etc.)
- Images are synthetic and well-structured
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image


class CLEVRDataset:
    """
    Dataset class for CLEVR (Compositional Language and Elementary Visual Reasoning).
    
    Loads images, questions, and answers from CLEVR dataset format.
    """
    
    def __init__(
        self,
        questions_path: str,
        images_dir: str,
        split: Optional[str] = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize CLEVR dataset.
        
        Args:
            questions_path: Path to CLEVR questions JSON file
            images_dir: Directory containing CLEVR images (subset)
            split: Dataset split ('train', 'val', 'test') - if None, uses all
            max_samples: Maximum number of samples to load (for quick testing)
        """
        self.questions_path = Path(questions_path)
        self.images_dir = Path(images_dir)
        self.split = split
        self.max_samples = max_samples
        
        # Load questions
        print(f"Loading CLEVR questions from {questions_path}...")
        with open(questions_path, 'r') as f:
            self.questions_data = json.load(f)
        
        # Prepare samples
        self.samples = self._prepare_samples()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} CLEVR samples")
    
    def _prepare_samples(self) -> List[Dict]:
        """
        Prepare list of samples from loaded questions data.
        
        Returns:
            List of sample dictionaries
        """
        samples = []
        questions = self.questions_data.get('questions', [])
        
        for q_data in questions:
            # Extract question info
            question = q_data.get('question', '')
            answer = q_data.get('answer', '')
            image_filename = q_data.get('image_filename', '')
            split_name = q_data.get('split', 'train')
            
            # Filter by split if specified
            if self.split and split_name != self.split:
                continue
            
            # Check if image exists in images_subset directory
            image_path = self.images_dir / image_filename
            if not image_path.exists():
                # Skip if image doesn't exist in subset
                continue
            
            sample = {
                'question_id': q_data.get('question_index', len(samples)),
                'image_filename': image_filename,
                'image_path': str(image_path),
                'question': question,
                'answer': answer,
                'split': split_name,
                'question_family_index': q_data.get('question_family_index', None)
            }
            
            samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - image: PIL Image
                - question: str
                - answer: str
                - image_id: str (filename)
                - question_id: int
        """
        sample = self.samples[idx]
        
        # Load image
        image_path = Path(sample['image_path'])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        
        return {
            'image': image,
            'question': sample['question'],
            'answer': sample['answer'],
            'question_id': sample['question_id'],
            'image_id': sample['image_filename'],
            'image_filename': sample['image_filename'],
            'split': sample['split']
        }
    
    def get_train_split(self) -> List[Dict]:
        """Get training split samples."""
        return [self[i] for i, s in enumerate(self.samples) if s['split'] == 'train']
    
    def get_val_split(self) -> List[Dict]:
        """Get validation split samples."""
        return [self[i] for i, s in enumerate(self.samples) if s['split'] == 'val']
    
    def get_test_split(self) -> List[Dict]:
        """Get test split samples."""
        return [self[i] for i, s in enumerate(self.samples) if s['split'] == 'test']


def load_clevr_subset(
    questions_path: str,
    images_dir: str,
    split: str = "val",
    num_samples: int = 200
) -> List[Dict]:
    """
    Load a small subset of CLEVR for quick evaluation.
    
    Args:
        questions_path: Path to CLEVR questions JSON file
        images_dir: Directory containing CLEVR images (subset)
        split: Dataset split ('train', 'val', 'test')
        num_samples: Number of samples to load
        
    Returns:
        List of sample dictionaries
    """
    dataset = CLEVRDataset(
        questions_path=questions_path,
        images_dir=images_dir,
        split=split,
        max_samples=num_samples
    )
    
    return [dataset[i] for i in range(len(dataset))]


def download_clevr_data(
    data_dir: str = "./data",
    download_images: bool = False
) -> Dict[str, str]:
    """
    Download CLEVR dataset files (instructions only - actual download requires manual steps).
    
    Args:
        data_dir: Directory to store data
        download_images: Whether to download images (large, 18GB full set)
        
    Returns:
        Dictionary with paths to downloaded files
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CLEVR Dataset Download Instructions")
    print("=" * 60)
    print("\n1. Download CLEVR v1.0 (no images) - 86 MB:")
    print("   https://cs.stanford.edu/people/jcjohns/clevr/CLEVR_v1.0_no_images.zip")
    print("\n2. Extract and place questions JSON file:")
    print(f"   Extract CLEVR_v1.0/questions/CLEVR_val_questions.json")
    print(f"   Place in: {data_path / 'clevr_questions.json'}")
    print("\n3. Download CLEVR images (OPTIONAL - use small subset):")
    print("   Full set: 18GB")
    print("   Recommended: Download only 100-300 images for testing")
    print("   Place images in: data/images_subset/")
    print("\n   Image naming format:")
    print("   CLEVR_train_000000.png")
    print("   CLEVR_val_000000.png")
    print("   CLEVR_test_000000.png")
    print("\n4. Structure should be:")
    print(f"   {data_path}/")
    print(f"     clevr_questions.json")
    print(f"     images_subset/")
    print(f"       CLEVR_val_000000.png")
    print(f"       CLEVR_val_000001.png")
    print(f"       ...")
    print("=" * 60)
    
    return {
        'questions': str(data_path / "clevr_questions.json"),
        'images': str(data_path / "images_subset")
    }


def load_clevr_dataset(
    questions_path: str = "data/clevr_questions.json",
    images_dir: str = "data/images_subset",
    split: str = "val",
    num_samples: Optional[int] = None
) -> List[Dict]:
    """
    Load CLEVR dataset from questions JSON file.
    
    This is the main function to use for loading CLEVR data.
    Alias for load_clevr_from_questions_file for convenience.
    
    Args:
        questions_path: Path to CLEVR questions JSON (e.g., CLEVR_val_questions.json)
        images_dir: Directory containing CLEVR images
        split: Dataset split ('train', 'val', 'test')
        num_samples: Maximum number of samples to load
        
    Returns:
        List of sample dictionaries with format:
        {
            'image': PIL Image,
            'question': str,
            'answer': str,
            'image_filename': str,
            'image_id': str,
            'question_id': int
        }
    
    Example:
        >>> from src.dataset_clevr import load_clevr_dataset
        >>> samples = load_clevr_dataset(split="val", num_samples=100)
    """
    return load_clevr_from_questions_file(questions_path, images_dir, split, num_samples)


def get_clevr_split(
    questions_path: str = "data/clevr_questions.json",
    images_dir: str = "data/images_subset",
    split: str = "val"
) -> List[Dict]:
    """
    Get a specific CLEVR split.
    
    Args:
        questions_path: Path to CLEVR questions JSON
        images_dir: Directory containing CLEVR images
        split: Dataset split ('train', 'val', 'test')
        
    Returns:
        List of sample dictionaries for the specified split
    """
    dataset = CLEVRDataset(
        questions_path=questions_path,
        images_dir=images_dir,
        split=split,
        max_samples=None  # Get all samples for this split
    )
    
    return [dataset[i] for i in range(len(dataset))]


def load_clevr_from_questions_file(
    questions_path: str,
    images_dir: str,
    split: str = "val",
    num_samples: Optional[int] = None
) -> List[Dict]:
    """
    Load CLEVR dataset from questions JSON file.
    
    This is the main function to use for loading CLEVR data.
    
    Args:
        questions_path: Path to CLEVR questions JSON (e.g., CLEVR_val_questions.json)
        images_dir: Directory containing CLEVR images
        split: Dataset split ('train', 'val', 'test')
        num_samples: Maximum number of samples to load
        
    Returns:
        List of sample dictionaries with format:
        {
            'image': PIL Image,
            'question': str,
            'answer': str,
            'image_id': str,
            'question_id': int,
            'image_filename': str
        }
    
    Example:
        >>> from src.dataset_clevr import load_clevr_from_questions_file
        >>> samples = load_clevr_from_questions_file(
        ...     questions_path="data/clevr_questions.json",
        ...     images_dir="data/images_subset",
        ...     split="val",
        ...     num_samples=100
        ... )
    """
    dataset = CLEVRDataset(
        questions_path=questions_path,
        images_dir=images_dir,
        split=split,
        max_samples=num_samples
    )
    
    return [dataset[i] for i in range(len(dataset))]
