"""
Text recognition using TrOCR models
"""

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np
import cv2
import re
from typing import Tuple, Optional

class TextRecognizer:
    """Handles text recognition using TrOCR models"""
    
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten"):
        self.model_name = model_name
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load TrOCR model and processor"""
        try:
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            print(f"✅ Loaded TrOCR model: {self.model_name}")
        except Exception as e:
            print(f"❌ Error loading TrOCR model: {e}")
            raise
    
    def recognize(self, image: np.ndarray, field_type: str = "text") -> Tuple[str, float]:
        """
        Recognize text in image patch
        
        Args:
            image: Input image patch
            field_type: Type of field (text, digits, currency, date)
            
        Returns:
            Tuple of (recognized_text, confidence_score)
        """
        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image).convert('RGB')
            
            # Generate text using TrOCR
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
            
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values, max_length=50)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Calculate confidence (simplified - could be improved)
            confidence = self._calculate_confidence(generated_text, field_type)
            
            # Apply field-specific postprocessing
            processed_text = self._postprocess_by_field_type(generated_text, field_type)
            
            return processed_text, confidence
            
        except Exception as e:
            print(f"Error in text recognition: {e}")
            return "", 0.0
    
    def _calculate_confidence(self, text: str, field_type: str) -> float:
        """Calculate confidence score based on text and field type"""
        if not text.strip():
            return 0.0
        
        # Base confidence
        confidence = 0.5
        
        # Length-based adjustment
        if len(text.strip()) > 2:
            confidence += 0.2
        
        # Field-type specific patterns
        if field_type == "currency":
            if re.search(r'\$[\d,]+\.?\d*', text):
                confidence += 0.3
        elif field_type == "date":
            if re.search(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', text):
                confidence += 0.3
        elif field_type == "digits":
            if text.strip().replace(',', '').replace('.', '').isdigit():
                confidence += 0.3
        elif field_type == "text":
            if re.search(r'^[A-Za-z\s]+$', text.strip()):
                confidence += 0.2
        
        # Character quality
        clean_chars = len(re.findall(r'[A-Za-z0-9]', text))
        total_chars = len(text.strip())
        if total_chars > 0:
            char_ratio = clean_chars / total_chars
            confidence += char_ratio * 0.2
        
        return min(1.0, confidence)
    
    def _postprocess_by_field_type(self, text: str, field_type: str) -> str:
        """Apply field-specific postprocessing"""
        if field_type == "currency":
            return self._postprocess_currency(text)
        elif field_type == "date":
            return self._postprocess_date(text)
        elif field_type == "digits":
            return self._postprocess_digits(text)
        elif field_type == "text":
            return self._postprocess_text(text)
        else:
            return text.strip()
    
    def _postprocess_currency(self, text: str) -> str:
        """Postprocess currency fields"""
        # Remove common OCR errors and standardize format
        text = text.replace('O', '0').replace('l', '1').replace('I', '1')
        
        # Extract currency pattern
        currency_match = re.search(r'\$?[\d,]+\.?\d*', text)
        if currency_match:
            amount = currency_match.group()
            if not amount.startswith('$'):
                amount = '$' + amount
            return amount
        
        return text.strip()
    
    def _postprocess_date(self, text: str) -> str:
        """Postprocess date fields"""
        # Common OCR corrections
        text = text.replace('O', '0').replace('l', '1').replace('I', '1')
        
        # Extract date pattern
        date_match = re.search(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', text)
        if date_match:
            return date_match.group()
        
        return text.strip()
    
    def _postprocess_digits(self, text: str) -> str:
        """Postprocess digit-only fields"""
        # Extract only digits and common separators
        cleaned = re.sub(r'[^0-9,.]', '', text)
        return cleaned
    
    def _postprocess_text(self, text: str) -> str:
        """Postprocess text fields (names, etc.)"""
        # Remove template noise words
        template_words = {
            'guest', 'name', 'date', 'amount', 'total', 'signature',
            'form', 'rental', 'property', 'address', 'phone', 'email'
        }
        
        words = text.split()
        
        # Filter out template words and keep name-like words
        name_words = []
        for word in words:
            clean_word = re.sub(r'[^A-Za-z]', '', word)
            if (len(clean_word) >= 2 and 
                clean_word.lower() not in template_words and
                re.match(r'^[A-Z][a-z]+$', clean_word)):
                name_words.append(clean_word)
        
        # Keep up to 3 name components
        return ' '.join(name_words[:3]) if name_words else text.strip()
    
    def update_model(self, model_path: str):
        """Update to use a fine-tuned model"""
        try:
            self.model_name = model_path
            self._load_model()
            print(f"✅ Updated to model: {model_path}")
        except Exception as e:
            print(f"❌ Error updating model: {e}")
            raise