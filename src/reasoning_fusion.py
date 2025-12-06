"""
Novel Visual + Text Chain-of-Thought Fusion

This module implements the novel extension that fuses visual observations
and textual reasoning at each step.
"""

import torch
from typing import Dict, List, Tuple, Optional
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from .visual_utils import VisualRegionDetector
import re


class VisualTextFusionReasoner:
    """
    Novel visual + text fused chain-of-thought reasoner.
    """
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        num_fusion_steps: int = 4
    ):
        """
        Initialize visual-text fusion reasoner.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on
            load_in_4bit: Use 4-bit quantization
            load_in_8bit: Use 8-bit quantization
            num_fusion_steps: Number of fusion reasoning steps
        """
        self.device = device
        self.model_name = model_name
        self.num_fusion_steps = num_fusion_steps
        
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
        Perform visual + text fused chain-of-thought reasoning.
        
        Args:
            image: PIL Image
            question: Question to answer
            max_new_tokens: Maximum tokens per step
            temperature: Sampling temperature
            
        Returns:
            Dictionary with:
                - visual_steps: List of visual observations
                - text_steps: List of textual reasoning steps
                - fused_steps: List of combined steps
                - answer: Final answer
        """
        # Step 1: Detect relevant visual regions
        print(f"Detecting relevant regions for question: {question[:50]}...")
        regions = self.region_detector.detect_relevant_regions_clip(
            image=image,
            question=question,
            num_regions=self.num_fusion_steps
        )
        
        visual_steps = []
        text_steps = []
        fused_steps = []
        conversation_memory = []
        
        # Step 2: Generate fused reasoning steps
        for step_idx, (region_crop, bbox, similarity) in enumerate(regions):
            print(f"Processing fusion step {step_idx + 1}/{self.num_fusion_steps}...")
            
            # A. Visual observation step
            visual_prompt = (
                f"USER: <image>\n"
                f"First, visually describe this image region in detail. "
                f"Focus on objects, their positions, colors, sizes, and spatial relationships. "
                f"Be specific and thorough.\n"
                f"ASSISTANT:"
            )
            
            inputs_visual = self.processor(
                text=visual_prompt,
                images=region_crop,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs_visual = self.model.generate(
                    **inputs_visual,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            visual_response = self.processor.decode(
                outputs_visual[0],
                skip_special_tokens=True
            )
            
            if "ASSISTANT:" in visual_response:
                visual_obs = visual_response.split("ASSISTANT:")[-1].strip()
            else:
                visual_obs = visual_response
            
            visual_steps.append({
                "step_number": step_idx + 1,
                "region_bbox": bbox,
                "similarity_score": similarity,
                "observation": visual_obs
            })
            
            # B. Textual reasoning step (integrating visual observation)
            # Build context from previous steps
            context_parts = []
            if conversation_memory:
                context_parts.append("Previous reasoning:")
                context_parts.extend(conversation_memory)
            
            context_parts.append(f"Current visual observation: {visual_obs}")
            
            text_prompt = (
                f"USER: <image>\n"
                f"Question: {question}\n\n"
                + "\n".join(context_parts) + "\n\n"
                f"Now reason about this visual observation in language. "
                f"What inference can you make? How does this relate to the question? "
                f"How does it integrate with previous steps?\n"
                f"ASSISTANT:"
            )
            
            # Use full image for textual reasoning (to maintain context)
            inputs_text = self.processor(
                text=text_prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs_text = self.model.generate(
                    **inputs_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            text_response = self.processor.decode(
                outputs_text[0],
                skip_special_tokens=True
            )
            
            if "ASSISTANT:" in text_response:
                text_reasoning = text_response.split("ASSISTANT:")[-1].strip()
            else:
                text_reasoning = text_response
            
            text_steps.append({
                "step_number": step_idx + 1,
                "reasoning": text_reasoning
            })
            
            # C. Create fused step
            fused_step = {
                "step_number": step_idx + 1,
                "visual_observation": visual_obs,
                "textual_reasoning": text_reasoning,
                "region_bbox": bbox
            }
            fused_steps.append(fused_step)
            
            # Update conversation memory
            conversation_memory.append(
                f"Step {step_idx + 1} - Visual: {visual_obs[:100]}... "
                f"Reasoning: {text_reasoning[:100]}..."
            )
        
        # Step 3: Generate final answer using all fused steps
        print("Generating final answer from fused reasoning steps...")
        final_prompt = (
            f"USER: <image>\n"
            f"Question: {question}\n\n"
            f"Fused reasoning steps:\n"
        )
        
        for fused_step in fused_steps:
            final_prompt += (
                f"\nStep {fused_step['step_number']}:\n"
                f"Visual observation: {fused_step['visual_observation']}\n"
                f"Textual reasoning: {fused_step['textual_reasoning']}\n"
            )
        
        final_prompt += (
            f"\nBased on all the fused visual and textual reasoning steps above, "
            f"provide the final answer to the question.\n"
            f"ASSISTANT:"
        )
        
        inputs_final = self.processor(
            text=final_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs_final = self.model.generate(
                **inputs_final,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        final_response = self.processor.decode(
            outputs_final[0],
            skip_special_tokens=True
        )
        
        if "ASSISTANT:" in final_response:
            final_text = final_response.split("ASSISTANT:")[-1].strip()
        else:
            final_text = final_response
        
        # Extract answer
        answer = self._extract_answer(final_text)
        
        return {
            "visual_steps": [step["observation"] for step in visual_steps],
            "text_steps": [step["reasoning"] for step in text_steps],
            "fused_steps": fused_steps,
            "answer": answer,
            "regions": regions,
            "full_response": final_text,
            "method": "visual_text_fusion"
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


def coct_visual_text_fused(
    image: Image.Image,
    question: str,
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    device: str = "cuda",
    num_steps: int = 4
) -> Dict[str, any]:
    """
    Convenience function for visual + text fused reasoning.
    
    Args:
        image: PIL Image
        question: Question to answer
        model_name: Model identifier
        device: Device to run on
        num_steps: Number of fusion steps
        
    Returns:
        Dictionary with visual_steps, text_steps, fused_steps, and answer
    """
    reasoner = VisualTextFusionReasoner(
        model_name=model_name,
        device=device,
        num_fusion_steps=num_steps
    )
    return reasoner.reason(image, question)
