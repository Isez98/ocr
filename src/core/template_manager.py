"""
Template management and configuration
"""

import json
import os
from typing import Dict, List, Optional

class TemplateManager:
    """Manages template configurations and operations"""
    
    def __init__(self, templates_dir: str = None):
        self.templates_dir = templates_dir or os.path.join(os.getcwd(), "data", "templates")
    
    def load_template(self, template_id: str) -> Optional[Dict]:
        """Load template configuration"""
        template_path = os.path.join(self.templates_dir, f"{template_id}.json")
        
        if not os.path.exists(template_path):
            return None
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading template {template_id}: {e}")
            return None
    
    def save_template(self, template_id: str, config: Dict) -> bool:
        """Save template configuration"""
        template_path = os.path.join(self.templates_dir, f"{template_id}.json")
        
        try:
            os.makedirs(self.templates_dir, exist_ok=True)
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving template {template_id}: {e}")
            return False
    
    def list_templates(self) -> List[str]:
        """List available templates"""
        if not os.path.exists(self.templates_dir):
            return []
        
        templates = []
        for file in os.listdir(self.templates_dir):
            if file.endswith('.json'):
                templates.append(file[:-5])  # Remove .json extension
        
        return sorted(templates)
    
    def validate_template(self, config: Dict) -> List[str]:
        """Validate template configuration"""
        errors = []
        
        # Required fields
        required_fields = ['id', 'canonical_image', 'rois']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate ROIs
        if 'rois' in config:
            for i, roi in enumerate(config['rois']):
                roi_errors = self._validate_roi(roi, i)
                errors.extend(roi_errors)
        
        # Check canonical image exists
        if 'canonical_image' in config:
            canon_path = os.path.join(self.templates_dir, config['canonical_image'])
            if not os.path.exists(canon_path):
                errors.append(f"Canonical image not found: {config['canonical_image']}")
        
        return errors
    
    def _validate_roi(self, roi: Dict, index: int) -> List[str]:
        """Validate a single ROI configuration"""
        errors = []
        
        # Required ROI fields
        required_fields = ['name', 'x', 'y', 'w', 'h']
        for field in required_fields:
            if field not in roi:
                errors.append(f"ROI {index}: Missing required field '{field}'")
        
        # Validate coordinates
        for coord in ['x', 'y', 'w', 'h']:
            if coord in roi:
                try:
                    value = float(roi[coord])
                    if coord in ['w', 'h'] and value <= 0:
                        errors.append(f"ROI {index}: {coord} must be positive")
                    elif coord in ['x', 'y'] and value < 0:
                        errors.append(f"ROI {index}: {coord} must be non-negative")
                except (ValueError, TypeError):
                    errors.append(f"ROI {index}: {coord} must be a number")
        
        return errors
    
    def get_template_info(self, template_id: str) -> Optional[Dict]:
        """Get template metadata and statistics"""
        config = self.load_template(template_id)
        if not config:
            return None
        
        return {
            "id": config.get("id", template_id),
            "name": config.get("name", template_id),
            "description": config.get("description", ""),
            "roi_count": len(config.get("rois", [])),
            "field_types": self._get_field_types(config),
            "canvas_size": {
                "width": config.get("canvas_width"),
                "height": config.get("canvas_height")
            },
            "validation_errors": self.validate_template(config)
        }
    
    def _get_field_types(self, config: Dict) -> Dict[str, int]:
        """Get count of field types in template"""
        field_types = {}
        for roi in config.get("rois", []):
            field_type = roi.get("type", "text")
            field_types[field_type] = field_types.get(field_type, 0) + 1
        return field_types