"""
Original CoVT-Style Visual Chain-of-Thought Reasoning

This module implements the original Chain-of-Visual-Thought approach
where reasoning proceeds through visual steps (describing image regions).
"""

import torch
from typing import Dict, List, Tuple, Optional
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from .visual_utils import VisualRegionDetector
import re


class CoVTVisualReasoner:
    """
    CoVT-style visual chain-of-thought reasoner.
    """
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        num_visual_steps: int = 3
    ):
        """
        Initialize CoVT visual reasoner.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on
            load_in_4bit: Use 4-bit quantization
            load_in_8bit: Use 8-bit quantization
            num_visual_steps: Number of visual reasoning steps
        """
        self.device = device
        self.model_name = model_name
        self.num_visual_steps = num_visual_steps
        
        print(f"Loading LLaVA model: {model_name}")
        
        # Load model with optional quantization
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        elif load_in_8bit:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        
        self.processor = LlavaProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # Initialize region detector
        self.region_detector = VisualRegionDetector(device=device)
        
        print("Model loaded successfully")
    
    def reason(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> Dict[str, any]:
        """
        Perform CoVT-style visual chain-of-thought reasoning.
        
        Args:
            image: PIL Image
            question: Question to answer
            max_new_tokens: Maximum tokens per step
            temperature: Sampling temperature
            
        Returns:
            Dictionary with:
                - steps: List of visual reasoning steps
                - answer: Final answer
                - regions: Detected regions used
        """
        # Step 1: Detect relevant visual regions
        print(f"Detecting relevant regions for question: {question[:50]}...")
        regions = self.region_detector.detect_relevant_regions_clip(
            image=image,
            question=question,
            num_regions=self.num_visual_steps
        )
        
        visual_steps = []
        conversation_history = []
        
        # Step 2: Generate visual reasoning steps
        for step_idx, (region_crop, bbox, similarity) in enumerate(regions):
            print(f"Processing visual step {step_idx + 1}/{self.num_visual_steps}...")
            
            # Generate prompt for this region
            region_prompt = self.region_detector.get_region_description_prompt(
                region_idx=step_idx,
                question=question
            )
            
            # Build conversation context
            if conversation_history:
                context = "\n".join(conversation_history)
                prompt = (
                    f"USER: <image>\n"
                    f"Previous observations:\n{context}\n\n"
                    f"{region_prompt}\n"
                    f"ASSISTANT:"
                )
            else:
                prompt = (
                    f"USER: <image>\n"
                    f"{region_prompt}\n"
                    f"ASSISTANT:"
                )
            
            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=region_crop,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.processor.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            if "ASSISTANT:" in generated_text:
                step_response = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                step_response = generated_text
            
            visual_steps.append({
                "step_number": step_idx + 1,
                "region_bbox": bbox,
                "similarity_score": similarity,
                "description": step_response
            })
            
            conversation_history.append(
                f"Step {step_idx + 1}: {step_response}"
            )
        
        # Step 3: Generate final answer based on all visual steps
        print("Generating final answer from visual reasoning steps...")
        final_prompt = (
            f"USER: <image>\n"
            f"Question: {question}\n\n"
            f"Visual reasoning steps:\n" + "\n".join(conversation_history) + "\n\n"
            f"Based on your visual reasoning steps above, provide the final answer to the question.\n"
            f"ASSISTANT:"
        )
        
        # Use original full image for final answer
        inputs = self.processor(
            text=final_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        generated_text = self.processor.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        if "ASSISTANT:" in generated_text:
            final_response = generated_text.split("ASSISTANT:")[-1].strip()
        else:
            final_response = generated_text
        
        # Extract answer from final response
        answer = self._extract_answer(final_response)
        
        return {
            "steps": [step["description"] for step in visual_steps],
            "answer": answer,
            "visual_steps": visual_steps,
            "regions": regions,
            "full_response": final_response,
            "method": "covt_visual_cot"
        }
    
    def _extract_answer(self, response: str) -> str:
        """
        Extract final answer from response text.
        
        Args:
            response: Model response text
            
        Returns:
            Extracted answer string
        """
        # Try to find explicit answer markers
        answer_patterns = [
            r'(?:final answer|answer is|the answer)[:\.]\s*(.+?)(?:\.|$)',
            r'Therefore[^.]*\.\s*(.+?)(?:\.|$)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                return answer.strip('.,!?')
        
        # If no explicit answer, use last sentence
        sentences = response.split('.')
        answer = sentences[-1].strip() if sentences else response.strip()
        return answer.strip('.,!?')


def coct_visual_reason(
    image: Image.Image,
    question: str,
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    device: str = "cuda",
    num_steps: int = 3
) -> Dict[str, any]:
    """
    Convenience function for CoVT-style visual reasoning.
    
    Args:
        image: PIL Image
        question: Question to answer
        model_name: Model identifier
        device: Device to run on
        num_steps: Number of visual reasoning steps
        
    Returns:
        Dictionary with steps, answer, and metadata
    """
    reasoner = CoVTVisualReasoner(
        model_name=model_name,
        device=device,
        num_visual_steps=num_steps
    )
    return reasoner.reason(image, question)
