#!/usr/bin/env python3
"""
Test script for snap-to-ink functionality with synthetic data.
"""

import cv2
import numpy as np
import json
import tempfile
import os
from snap_to_ink import snap_template_to_ink, snap_roi_to_ink


def create_test_image_and_template():
    """Create a synthetic test image and template for demonstration."""
    
    # Create a 800x600 test image with white background
    height, width = 600, 800
    image = np.ones((height, width), dtype=np.uint8) * 255
    
    # Add some text-like regions with black ink
    
    # Region 1: Name field (text) - loose box around text
    cv2.rectangle(image, (100, 100), (300, 130), 0, -1)  # Solid text block
    roi1 = {"name": "name", "x": 80, "y": 80, "w": 240, "h": 70, "type": "text"}
    
    # Region 2: Phone number (digits) - scattered digits
    for i, x in enumerate([100, 120, 140, 170, 190, 210, 240, 260, 280, 300]):
        cv2.rectangle(image, (x, 200), (x+15, 220), 0, -1)
    roi2 = {"name": "phone", "x": 80, "y": 180, "w": 250, "h": 60, "type": "phone"}
    
    # Region 3: Currency amount - with decimal point
    cv2.rectangle(image, (100, 300), (120, 320), 0, -1)  # $
    cv2.rectangle(image, (130, 300), (180, 320), 0, -1)  # digits
    cv2.rectangle(image, (185, 315), (190, 320), 0, -1)  # decimal point
    cv2.rectangle(image, (195, 300), (220, 320), 0, -1)  # cents
    roi3 = {"name": "amount", "x": 80, "y": 280, "w": 200, "h": 60, "type": "currency"}
    
    # Region 4: Date field
    cv2.rectangle(image, (100, 400), (130, 420), 0, -1)  # month
    cv2.rectangle(image, (140, 410), (145, 415), 0, -1)  # /
    cv2.rectangle(image, (150, 400), (180, 420), 0, -1)  # day
    cv2.rectangle(image, (190, 410), (195, 415), 0, -1)  # /
    cv2.rectangle(image, (200, 400), (250, 420), 0, -1)  # year
    roi4 = {"name": "date", "x": 80, "y": 380, "w": 200, "h": 60, "type": "date"}
    
    # Create template
    template = {
        "id": "test_template",
        "canvas_width": width,
        "canvas_height": height,
        "canonical_image": "test_image.png",
        "rois": [roi1, roi2, roi3, roi4]
    }
    
    return image, template


def test_snap_to_ink():
    """Test the snap-to-ink functionality."""
    
    print("=== Testing Snap-to-Ink Functionality ===\\n")
    
    # Create test data
    image, template = create_test_image_and_template()
    
    # Save to temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir, "test_image.png")
        template_path = os.path.join(temp_dir, "test_template.json")
        
        cv2.imwrite(image_path, image)
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"Created test image: {image_path}")
        print(f"Created test template: {template_path}\\n")
        
        # Test individual ROI refinement
        print("Testing individual ROI refinement:")
        gray_image = image
        for i, roi in enumerate(template["rois"]):
            print(f"\\nROI {i+1}: {roi['name']} ({roi['type']})")
            print(f"  Original: x={roi['x']}, y={roi['y']}, w={roi['w']}, h={roi['h']}")
            
            refined_roi = snap_roi_to_ink(gray_image, roi, template["canvas_width"], template["canvas_height"])
            print(f"  Refined:  x={refined_roi['x']}, y={refined_roi['y']}, w={refined_roi['w']}, h={refined_roi['h']}")
            
            # Calculate reduction in area
            orig_area = roi['w'] * roi['h']
            refined_area = refined_roi['w'] * refined_roi['h']
            reduction = (orig_area - refined_area) / orig_area * 100
            print(f"  Area reduction: {reduction:.1f}%")
        
        # Test full template refinement with preview
        print("\\n" + "="*50)
        print("Testing full template refinement:")
        
        try:
            refined_template, preview_image = snap_template_to_ink(
                image_path, template, preview_mode=True
            )
            
            print(f"\\nRefined template has {len(refined_template['rois'])} ROIs")
            
            # Save preview image
            preview_path = os.path.join(temp_dir, "preview.png")
            cv2.imwrite(preview_path, preview_image)
            print(f"Preview image saved: {preview_path}")
            
            # Show summary
            print("\\nSummary of changes:")
            total_original_area = 0
            total_refined_area = 0
            
            for i, (original, refined) in enumerate(zip(template["rois"], refined_template["rois"])):
                orig_area = original['w'] * original['h']
                refined_area = refined['w'] * refined['h']
                total_original_area += orig_area
                total_refined_area += refined_area
                
                if (original['x'] != refined['x'] or original['y'] != refined['y'] or
                    original['w'] != refined['w'] or original['h'] != refined['h']):
                    reduction = (orig_area - refined_area) / orig_area * 100
                    print(f"  {original['name']}: {reduction:.1f}% area reduction")
                else:
                    print(f"  {original['name']}: No change")
            
            overall_reduction = (total_original_area - total_refined_area) / total_original_area * 100
            print(f"\\nOverall area reduction: {overall_reduction:.1f}%")
            
            # Display preview (if running in an environment that supports it)
            try:
                cv2.imshow('Snap-to-Ink Test (Red=Original, Green=Refined)', preview_image)
                print("\\nPreview window opened. Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except:
                print("\\nPreview display not available in this environment")
                
        except Exception as e:
            print(f"Error during template processing: {e}")
            import traceback
            traceback.print_exc()


def test_padding_logic():
    """Test the type-specific padding logic."""
    
    print("\\n" + "="*50)
    print("Testing type-specific padding logic:")
    
    from snap_to_ink import get_type_specific_padding
    
    canvas_width, canvas_height = 1000, 800
    
    types = ["text", "digits", "currency", "date", "phone"]
    
    for roi_type in types:
        padding = get_type_specific_padding(roi_type, canvas_width, canvas_height)
        print(f"\\n{roi_type.upper()}:")
        print(f"  Top: {padding['top']}px, Bottom: {padding['bottom']}px")
        print(f"  Left: {padding['left']}px, Right: {padding['right']}px")


if __name__ == "__main__":
    try:
        test_snap_to_ink()
        test_padding_logic()
        print("\\n=== All tests completed successfully! ===")
    except Exception as e:
        print(f"\\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()