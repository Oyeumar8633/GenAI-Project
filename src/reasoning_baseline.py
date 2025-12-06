"""
Baseline Text-Only Chain-of-Thought Reasoning

This module implements the baseline text-only CoT reasoning using LLaVA,
without any visual chain-of-thought steps.
"""

import torch
from typing import Dict, List, Optional
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration
import re


class TextOnlyCoTReasoner:
    """
    Baseline text-only chain-of-thought reasoner using LLaVA.
    """
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False
    ):
        """
        Initialize text-only CoT reasoner.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on
            load_in_4bit: Use 4-bit quantization
            load_in_8bit: Use 8-bit quantization
        """
        self.device = device
        self.model_name = model_name
        
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
        
        print("Model loaded successfully")
    
    def reason(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ) -> Dict[str, any]:
        """
        Perform text-only chain-of-thought reasoning.
        
        Args:
            image: PIL Image
            question: Question to answer
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary with:
                - steps: List of reasoning steps (text)
                - answer: Final answer
                - full_response: Complete model response
        """
        # Construct prompt for text-only CoT
        prompt = (
            f"USER: <image>\n"
            f"Provide detailed step-by-step reasoning in TEXT ONLY to answer this question: {question}\n"
            f"Think through the problem step by step, then provide your final answer.\n"
            f"ASSISTANT:"
        )
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=image,
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
        
        # Extract assistant response
        if "ASSISTANT:" in generated_text:
            response = generated_text.split("ASSISTANT:")[-1].strip()
        else:
            response = generated_text
        
        # Parse reasoning steps and answer
        steps, answer = self._parse_cot_response(response, question)
        
        return {
            "steps": steps,
            "answer": answer,
            "full_response": response,
            "method": "text_only_cot"
        }
    
    def _parse_cot_response(
        self,
        response: str,
        question: str
    ) -> tuple[List[str], str]:
        """
        Parse CoT response into steps and final answer.
        
        Args:
            response: Model response text
            question: Original question
            
        Returns:
            Tuple of (steps_list, final_answer)
        """
        steps = []
        
        # Try to extract numbered steps
        step_patterns = [
            r'(?:Step \d+[:\.]|^\d+[\.\)])\s*(.+?)(?=(?:Step \d+|^\d+[\.\)]|$))',
            r'(?:First|Second|Third|Fourth|Fifth|Next|Then|Finally)[:\.]\s*(.+?)(?=(?:First|Second|Third|Fourth|Fifth|Next|Then|Finally|$))',
        ]
        
        for pattern in step_patterns:
            matches = re.finditer(pattern, response, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                step_text = match.group(1).strip()
                if len(step_text) > 10:  # Filter out very short matches
                    steps.append(step_text)
        
        # If no structured steps found, split by sentences
        if not steps:
            sentences = re.split(r'[.!?]\s+', response)
            steps = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Extract final answer (usually last sentence or after "answer:" / "final answer:")
        answer_patterns = [
            r'(?:final answer|answer is|the answer)[:\.]\s*(.+?)(?:\.|$)',
            r'Therefore[^.]*\.\s*(.+?)(?:\.|$)',
        ]
        
        answer = None
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                break
        
        # If no explicit answer found, use last sentence
        if not answer:
            sentences = response.split('.')
            answer = sentences[-1].strip() if sentences else response.strip()
        
        # Clean up answer
        answer = re.sub(r'^(the |a |an )', '', answer, flags=re.IGNORECASE)
        answer = answer.strip('.,!?')
        
        return steps, answer


def cot_text_only(
    image: Image.Image,
    question: str,
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    device: str = "cuda"
) -> Dict[str, any]:
    """
    Convenience function for text-only CoT reasoning.
    
    Args:
        image: PIL Image
        question: Question to answer
        model_name: Model identifier
        device: Device to run on
        
    Returns:
        Dictionary with steps, answer, and metadata
    """
    reasoner = TextOnlyCoTReasoner(model_name=model_name, device=device)
    return reasoner.reason(image, question)
