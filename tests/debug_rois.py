#!/usr/bin/env python3
"""
Debug tool to visualize current ROIs and test extraction
Usage: python debug_rois.py template.json image.jpg
"""

import cv2
import json
import sys
import os
import numpy as np
from PIL import Image

def load_template(template_path):
    with open(template_path, 'r') as f:
        return json.load(f)

def load_and_align_image(image_path, canonical_path):
    """Simplified alignment - you can use your existing alignment logic"""
    img = cv2.imread(image_path)
    canonical = cv2.imread(canonical_path)
    
    if img is None or canonical is None:
        return None, None
    
    # For debugging, just resize to canonical size
    # In production, use your alignment function
    h, w = canonical.shape[:2]
    aligned = cv2.resize(img, (w, h))
    
    return aligned, canonical

def crop_roi(img, roi):
    """Extract ROI from image"""
    x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
    return img[y:y+h, x:x+w]

def visualize_rois(template_path, image_path=None):
    template = load_template(template_path)
    
    # Load canonical image
    canonical_path = os.path.join(os.path.dirname(template_path), 
                                 os.path.basename(template['canonical_image']))
    
    if not os.path.exists(canonical_path):
        print(f"Canonical image not found: {canonical_path}")
        return
    
    canonical = cv2.imread(canonical_path)
    if canonical is None:
        print(f"Could not load canonical image: {canonical_path}")
        return
    
    # Use provided image or canonical
    if image_path and os.path.exists(image_path):
        test_img, _ = load_and_align_image(image_path, canonical_path)
        display_img = test_img.copy()
        window_title = f"ROI Visualization - {os.path.basename(image_path)}"
    else:
        display_img = canonical.copy()
        window_title = f"ROI Visualization - {os.path.basename(canonical_path)}"
    
    if display_img is None:
        print("Could not load image for visualization")
        return
    
    # Draw ROIs
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, roi in enumerate(template['rois']):
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
        color = colors[i % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 2)
        
        # Draw label with background
        label = f"{roi['name']} ({roi['type']})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Background rectangle for text
        cv2.rectangle(display_img, (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), color, -1)
        
        # Text
        cv2.putText(display_img, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show ROI contents in separate window
        if image_path:
            roi_img = crop_roi(test_img, roi)
            if roi_img.size > 0:
                # Resize for better visibility
                scale = max(1, 200 // max(roi_img.shape[:2]))
                roi_resized = cv2.resize(roi_img, None, fx=scale, fy=scale, 
                                       interpolation=cv2.INTER_NEAREST)
                cv2.imshow(f"ROI: {roi['name']}", roi_resized)
    
    # Scale display if too large
    height, width = display_img.shape[:2]
    if height > 800:
        scale = 800 / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        display_img = cv2.resize(display_img, (new_width, new_height))
    
    cv2.imshow(window_title, display_img)
    
    print(f"Showing ROIs for template: {template['id']}")
    print(f"Canvas size: {template['canvas_width']} x {template['canvas_height']}")
    print("\nROI Details:")
    for i, roi in enumerate(template['rois']):
        print(f"{i+1}. {roi['name']}: {roi['x']},{roi['y']} {roi['w']}x{roi['h']} ({roi['type']})")
    
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_rois.py <template.json> [image.jpg]")
        sys.exit(1)
    
    template_path = sys.argv[1]
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(template_path):
        print(f"Template file not found: {template_path}")
        sys.exit(1)
    
    if image_path and not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        sys.exit(1)
    
    visualize_rois(template_path, image_path)

if __name__ == "__main__":
    main()