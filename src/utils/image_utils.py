"""
Common utility functions for OCR processing
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict
import json
import os

def rel_to_px(roi: Dict, canvas_w: int, canvas_h: int, image_w: int, image_h: int, 
              field_type: str = "text") -> Tuple[int, int, int, int]:
    """
    Convert relative ROI coordinates to pixel coordinates with adaptive padding
    
    Args:
        roi: ROI configuration with x, y, w, h coordinates
        canvas_w, canvas_h: Template canvas dimensions
        image_w, image_h: Actual image dimensions
        field_type: Type of field for adaptive padding
        
    Returns:
        Tuple of (x, y, width, height) in pixels
    """
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    
    # Calculate adaptive padding
    padding_factor = get_adaptive_padding(canvas_w, canvas_h, field_type)
    
    # Convert to pixel coordinates
    if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
        # Relative coordinates
        px_x = int(x * image_w)
        px_y = int(y * image_h)
        px_w = int(w * image_w)
        px_h = int(h * image_h)
    else:
        # Already pixel coordinates
        px_x, px_y, px_w, px_h = int(x), int(y), int(w), int(h)
    
    # Apply adaptive padding
    px_x = max(0, px_x - padding_factor)
    px_y = max(0, px_y - padding_factor)
    px_w = min(image_w - px_x, px_w + 2 * padding_factor)
    px_h = min(image_h - px_y, px_h + 2 * padding_factor)
    
    return px_x, px_y, px_w, px_h

def get_adaptive_padding(canvas_w: int, canvas_h: int, field_type: str) -> int:
    """Calculate adaptive padding based on canvas size and field type"""
    base_padding = min(canvas_w, canvas_h) * 0.005  # 0.5% of smaller dimension
    
    type_multipliers = {
        "text": 1.5,
        "digits": 1.0,
        "currency": 1.2,
        "date": 1.0
    }
    
    multiplier = type_multipliers.get(field_type, 1.0)
    return max(2, int(base_padding * multiplier))

def crop_roi(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Safely crop ROI from image with bounds checking"""
    img_h, img_w = image.shape[:2]
    
    # Ensure coordinates are within image bounds
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    
    return image[y:y+h, x:x+w]

def validate_roi_coordinates(roi: Dict, canvas_w: int = None, canvas_h: int = None) -> List[str]:
    """Validate ROI coordinate values"""
    errors = []
    
    required_fields = ['x', 'y', 'w', 'h']
    for field in required_fields:
        if field not in roi:
            errors.append(f"Missing coordinate: {field}")
            continue
        
        try:
            value = float(roi[field])
            
            # Check for reasonable values
            if field in ['w', 'h'] and value <= 0:
                errors.append(f"{field} must be positive")
            elif field in ['x', 'y'] and value < 0:
                errors.append(f"{field} must be non-negative")
            
            # Check relative coordinate bounds
            if 0 <= value <= 1:
                if field in ['x', 'w'] and canvas_w and value > 1:
                    errors.append(f"{field} exceeds canvas width")
                elif field in ['y', 'h'] and canvas_h and value > 1:
                    errors.append(f"{field} exceeds canvas height")
                    
        except (ValueError, TypeError):
            errors.append(f"{field} must be a number")
    
    return errors

def load_json_config(file_path: str) -> Dict:
    """Safely load JSON configuration file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")

def save_json_config(data: Dict, file_path: str) -> None:
    """Safely save JSON configuration file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise IOError(f"Failed to save configuration to {file_path}: {e}")

def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Ensure image is in grayscale format"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure image is in RGB format"""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def calculate_image_stats(image: np.ndarray) -> Dict:
    """Calculate basic statistics about an image"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    return {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "mean_intensity": float(np.mean(gray)),
        "std_intensity": float(np.std(gray)),
        "min_intensity": int(np.min(gray)),
        "max_intensity": int(np.max(gray)),
        "total_pixels": int(gray.size)
    }