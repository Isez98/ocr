"""
Core OCR processing functionality
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image

from .image_processor import ImageProcessor
from .text_recognizer import TextRecognizer
from .template_manager import TemplateManager

class OCRProcessor:
    """Main OCR processing pipeline"""
    
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.text_recognizer = TextRecognizer()
        self.template_manager = TemplateManager()
    
    def process_form(self, image: np.ndarray, template_id: str) -> Dict:
        """
        Process a form image using the specified template
        
        Args:
            image: Input form image
            template_id: Template identifier
            
        Returns:
            Processing results with extracted field data
        """
        # Load template configuration
        template_config = self.template_manager.load_template(template_id)
        if not template_config:
            raise ValueError(f"Template {template_id} not found")
        
        # Align image to canonical template
        aligned_image = self.image_processor.align_to_template(image, template_config)
        
        # Extract text from ROIs
        results = []
        for roi in template_config["rois"]:
            field_result = self._process_roi(aligned_image, roi, template_config)
            results.append(field_result)
        
        return {
            "template_id": template_id,
            "fields": results,
            "processing_metadata": self._get_processing_metadata(results)
        }
    
    def _process_roi(self, image: np.ndarray, roi: Dict, template_config: Dict) -> Dict:
        """Process a single ROI"""
        # Extract ROI patch
        roi_patch = self.image_processor.extract_roi(image, roi, template_config)
        
        # Recognize text with multiple scales for robustness
        best_result = None
        best_confidence = 0.0
        
        scales = [1.0, 1.2, 1.5, 2.0]
        
        for scale in scales:
            if scale != 1.0:
                scaled_patch = self.image_processor.scale_image(roi_patch, scale)
            else:
                scaled_patch = roi_patch
            
            # Recognize text
            text, confidence = self.text_recognizer.recognize(
                scaled_patch, 
                field_type=roi.get("type", "text")
            )
            
            # Keep best result
            if confidence > best_confidence or (scale == 1.0 and confidence > 0.5):
                best_result = {
                    "name": roi["name"],
                    "text": text,
                    "confidence": round(float(confidence), 3),
                    "scale_used": scale,
                    "needs_review": confidence < 0.8,
                    "review_reason": f"Low confidence ({confidence:.2f})" if confidence < 0.8 else None
                }
                best_confidence = confidence
            
            # Early exit for good results
            if scale == 1.0 and confidence > 0.7:
                break
        
        return best_result or {
            "name": roi["name"],
            "text": "",
            "confidence": 0.0,
            "scale_used": 1.0,
            "needs_review": True,
            "review_reason": "No text detected"
        }
    
    def _get_processing_metadata(self, results: List[Dict]) -> Dict:
        """Generate processing metadata"""
        review_needed = [r for r in results if r.get("needs_review", False)]
        
        return {
            "total_fields": len(results),
            "fields_needing_review": len(review_needed),
            "review_fields": [
                {"name": r["name"], "reason": r.get("review_reason")} 
                for r in review_needed
            ],
            "average_confidence": np.mean([r["confidence"] for r in results if r["confidence"] > 0])
        }