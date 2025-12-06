"""
Visual Utilities for Image Processing and Region Detection

This module provides functions for:
- Image cropping and region extraction
- CLIP-based region relevance detection
- Attention map generation (GradCAM)
- Bounding box generation
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional, Dict
import torchvision.transforms as transforms

# Lazy import CLIP to avoid import errors at module level
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError as e:
    CLIP_AVAILABLE = False
    CLIP_IMPORT_ERROR = str(e)


class VisualRegionDetector:
    """
    Detects relevant visual regions in images for reasoning.
    """
    
    def __init__(self, device: str = "cuda", use_clip: bool = True):
        """
        Initialize region detector.
        
        Args:
            device: Device to run on ('cuda' or 'cpu')
            use_clip: Whether to use CLIP for region detection
        """
        self.device = device
        self.use_clip = use_clip
        
        if use_clip:
            if not CLIP_AVAILABLE:
                raise ImportError(
                    f"CLIP is not available. Original error: {CLIP_IMPORT_ERROR}\n"
                    "Please install compatible versions: pip install transformers>=4.35.0,<4.45.0 peft>=0.6.0"
                )
            print("Loading CLIP model for region detection...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(device)
            self.clip_model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def generate_grid_crops(
        self,
        image: Image.Image,
        grid_size: Tuple[int, int] = (3, 3),
        overlap: float = 0.1
    ) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Generate grid-based crops from image.
        
        Args:
            image: PIL Image
            grid_size: (rows, cols) for grid
            overlap: Overlap ratio between crops
            
        Returns:
            List of (crop_image, bbox) tuples where bbox is (x1, y1, x2, y2)
        """
        width, height = image.size
        rows, cols = grid_size
        
        crops = []
        
        # Calculate crop dimensions with overlap
        crop_w = int(width / cols * (1 + overlap))
        crop_h = int(height / rows * (1 + overlap))
        step_w = int(width / cols)
        step_h = int(height / rows)
        
        for i in range(rows):
            for j in range(cols):
                x1 = max(0, j * step_w - int(overlap * crop_w / 2))
                y1 = max(0, i * step_h - int(overlap * crop_h / 2))
                x2 = min(width, x1 + crop_w)
                y2 = min(height, y1 + crop_h)
                
                crop = image.crop((x1, y1, x2, y2))
                crops.append((crop, (x1, y1, x2, y2)))
        
        return crops
    
    def detect_relevant_regions_clip(
        self,
        image: Image.Image,
        question: str,
        num_regions: int = 5,
        grid_size: Tuple[int, int] = (3, 3)
    ) -> List[Tuple[Image.Image, Tuple[int, int, int, int], float]]:
        """
        Detect relevant regions using CLIP similarity.
        
        Args:
            image: PIL Image
            question: Question text
            num_regions: Number of top regions to return
            grid_size: Grid size for generating crops
            
        Returns:
            List of (crop_image, bbox, similarity_score) tuples
        """
        if not self.use_clip:
            # Fallback to grid crops
            crops = self.generate_grid_crops(image, grid_size)
            return [(crop, bbox, 0.5) for crop, bbox in crops[:num_regions]]
        
        # Generate grid crops
        crops = self.generate_grid_crops(image, grid_size)
        
        # Compute CLIP similarity for each crop
        similarities = []
        
        with torch.no_grad():
            # Encode question
            question_inputs = self.clip_processor(
                text=[question],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            question_embedding = self.clip_model.get_text_features(**question_inputs)
            question_embedding = F.normalize(question_embedding, dim=-1)
            
            # Encode each crop
            for crop, bbox in crops:
                crop_inputs = self.clip_processor(
                    images=crop,
                    return_tensors="pt"
                ).to(self.device)
                crop_embedding = self.clip_model.get_image_features(**crop_inputs)
                crop_embedding = F.normalize(crop_embedding, dim=-1)
                
                # Compute cosine similarity
                similarity = torch.matmul(question_embedding, crop_embedding.T).item()
                similarities.append((crop, bbox, similarity))
        
        # Sort by similarity and return top regions
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:num_regions]
    
    def detect_relevant_regions_attention(
        self,
        image: Image.Image,
        model,
        question: str,
        num_regions: int = 5
    ) -> List[Tuple[Image.Image, Tuple[int, int, int, int], float]]:
        """
        Detect relevant regions using attention maps (GradCAM).
        
        Args:
            image: PIL Image
            model: Vision-language model with attention
            question: Question text
            num_regions: Number of regions to return
            
        Returns:
            List of (crop_image, bbox, attention_score) tuples
        """
        # This is a placeholder - actual implementation would require
        # hooking into model's attention layers
        # For now, fallback to CLIP-based detection
        return self.detect_relevant_regions_clip(image, question, num_regions)
    
    def get_region_description_prompt(
        self,
        region_idx: int,
        question: str
    ) -> str:
        """
        Generate prompt for describing a visual region.
        
        Args:
            region_idx: Index of the region
            question: Original question
            
        Returns:
            Prompt string
        """
        return (
            f"Region {region_idx + 1}: Describe the visual content in this image region "
            f"that is relevant to answering the question: '{question}'. "
            f"Be specific about objects, their positions, colors, and relationships."
        )


def visualize_regions(
    image: Image.Image,
    regions: List[Tuple[Image.Image, Tuple[int, int, int, int], float]],
    save_path: Optional[str] = None
) -> Image.Image:
    """
    Visualize detected regions on the original image.
    
    Args:
        image: Original PIL Image
        regions: List of (crop, bbox, score) tuples
        save_path: Optional path to save visualization
        
    Returns:
        PIL Image with bounding boxes drawn
    """
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    for idx, (crop, bbox, score) in enumerate(regions):
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        color = (0, 255, 0) if score > 0.5 else (0, 0, 255)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"R{idx+1}: {score:.2f}"
        cv2.putText(img_cv, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    result = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    result_image = Image.fromarray(result)
    
    if save_path:
        result_image.save(save_path)
    
    return result_image


def prepare_image_for_model(
    image: Image.Image,
    target_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Prepare image tensor for model input.
    
    Args:
        image: PIL Image
        target_size: Target size (width, height)
        
    Returns:
        Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)
