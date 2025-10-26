#!/usr/bin/env python3
"""
Example script demonstrating enhanced OCR alignment capabilities.
Shows how to use fiducial markers and stable anchors for perfect warp correction.
"""

import cv2
import numpy as np
import json
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append('.')

try:
    from alignment_utils import (
        AlignmentEngine, 
        FiducialMarkerDetector, 
        StableAnchorDetector,
        create_test_fiducial_image
    )
    from snap_to_ink import snap_template_to_ink, validate_alignment_quality
    ALIGNMENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced alignment not available: {e}")
    ALIGNMENT_AVAILABLE = False


def create_example_template() -> dict:
    """Create an example template for demonstration."""
    return {
        "id": "demo_form",
        "canvas_width": 800,
        "canvas_height": 600,
        "canonical_image": "demo_canonical.png",
        "rois": [
            {
                "name": "name_field",
                "x": 200,
                "y": 150,
                "w": 300,
                "h": 40,
                "type": "text"
            },
            {
                "name": "date_field", 
                "x": 550,
                "y": 150,
                "w": 150,
                "h": 40,
                "type": "date"
            },
            {
                "name": "amount_field",
                "x": 200,
                "y": 250,
                "w": 200,
                "h": 40,
                "type": "currency"
            }
        ]
    }


def demo_fiducial_alignment():
    """Demonstrate fiducial marker-based alignment."""
    print("=== Fiducial Marker Alignment Demo ===\n")
    
    if not ALIGNMENT_AVAILABLE:
        print("Enhanced alignment not available. Please install opencv-contrib-python.")
        return
    
    # Create test image with fiducial markers
    test_image_path = "demo_fiducials.png"
    print(f"Creating test image with fiducial markers: {test_image_path}")
    
    create_test_fiducial_image(test_image_path, (800, 600), 50)
    
    # Create template
    template = create_example_template()
    template["canonical_image"] = test_image_path
    
    # Detect and add fiducials to template
    detector = FiducialMarkerDetector()
    image = cv2.imread(test_image_path)
    corners, ids, _ = detector.detect_markers(image)
    
    if ids is not None and len(ids) >= 4:
        centers = detector.get_marker_centers(corners, ids)
        template['fiducials'] = {int(marker_id): list(pos) for marker_id, pos in centers.items()}
        print(f"✓ Detected {len(centers)} fiducial markers: {list(centers.keys())}")
    else:
        print("✗ Could not detect sufficient fiducial markers")
        return
    
    # Save template
    template_path = "demo_template.json"
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    print(f"✓ Saved template: {template_path}")
    
    # Simulate a skewed/rotated input image
    print("\nSimulating skewed input image...")
    skewed_image_path = "demo_skewed.png"
    create_skewed_test_image(test_image_path, skewed_image_path)
    
    # Test alignment
    print("Testing alignment...")
    engine = AlignmentEngine()
    
    runtime_image = cv2.imread(skewed_image_path)
    canonical_points = {'fiducials': template['fiducials']}
    
    homography = engine.compute_alignment_homography(
        runtime_image, canonical_points, use_fiducials=True
    )
    
    if homography is not None:
        print("✓ Successfully computed homography matrix")
        
        # Apply alignment
        target_size = (template['canvas_width'], template['canvas_height'])
        aligned_image = engine.apply_alignment(runtime_image, homography, target_size)
        
        aligned_path = "demo_aligned.png"
        cv2.imwrite(aligned_path, aligned_image)
        print(f"✓ Saved aligned image: {aligned_path}")
        
        # Validate quality
        quality = validate_alignment_quality(runtime_image, aligned_image, template)
        print(f"✓ Alignment quality: {quality['overall_quality']:.2f}")
        
    else:
        print("✗ Alignment failed")
    
    print("\nFiducial alignment demo complete!")


def demo_anchor_alignment():
    """Demonstrate stable anchor-based alignment."""
    print("\n=== Stable Anchor Alignment Demo ===\n")
    
    if not ALIGNMENT_AVAILABLE:
        print("Enhanced alignment not available. Please install opencv-contrib-python.")
        return
    
    # Create a simple test image with geometric patterns
    test_image_path = "demo_anchors.png"
    print(f"Creating test image with stable anchors: {test_image_path}")
    
    create_anchor_test_image(test_image_path)
    
    # Create template
    template = create_example_template()
    template["canonical_image"] = test_image_path
    
    # Detect anchors
    image = cv2.imread(test_image_path)
    detector = StableAnchorDetector()
    anchors = detector.detect_all_anchors(image)
    
    if len(anchors) >= detector.min_anchors:
        # Add first few anchors to template
        anchor_dict = {}
        for i, (x, y) in enumerate(anchors[:5]):
            anchor_dict[f"anchor_{i}"] = [float(x), float(y)]
        
        template['anchors'] = anchor_dict
        print(f"✓ Detected {len(anchor_dict)} stable anchors")
        
        # Save template
        template_path = "demo_anchor_template.json"
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)
        print(f"✓ Saved template: {template_path}")
        
    else:
        print(f"✗ Insufficient anchors detected: {len(anchors)} < {detector.min_anchors}")
    
    print("\nAnchor alignment demo complete!")


def demo_snap_to_ink_with_alignment():
    """Demonstrate snap-to-ink processing with alignment."""
    print("\n=== Snap-to-Ink with Alignment Demo ===\n")
    
    if not ALIGNMENT_AVAILABLE:
        print("Enhanced alignment not available. Using basic snap-to-ink.")
        return
    
    # Use fiducial template if available
    template_path = "demo_template.json"
    image_path = "demo_skewed.png"
    
    if not Path(template_path).exists() or not Path(image_path).exists():
        print("Demo template or image not found. Run fiducial demo first.")
        return
    
    # Load template
    with open(template_path, 'r') as f:
        template = json.load(f)
    
    print("Processing with enhanced snap-to-ink...")
    
    # Process with alignment enabled
    try:
        refined_template, preview_image = snap_template_to_ink(
            image_path, template, 
            preview_mode=True,
            enable_alignment=True,
            use_fiducials=True
        )
        
        print("✓ Enhanced processing completed")
        
        if preview_image is not None:
            preview_path = "demo_snap_preview.png"
            cv2.imwrite(preview_path, preview_image)
            print(f"✓ Saved preview: {preview_path}")
        
        # Save refined template
        refined_path = "demo_template_refined.json"
        with open(refined_path, 'w') as f:
            json.dump(refined_template, f, indent=2)
        print(f"✓ Saved refined template: {refined_path}")
        
    except Exception as e:
        print(f"✗ Processing failed: {e}")
    
    print("\nSnap-to-ink with alignment demo complete!")


def create_skewed_test_image(input_path: str, output_path: str):
    """Create a skewed version of input image for testing."""
    image = cv2.imread(input_path)
    height, width = image.shape[:2]
    
    # Define skewing transformation
    src_points = np.float32([
        [0, 0],
        [width-1, 0],
        [0, height-1],
        [width-1, height-1]
    ])
    
    # Add some skew and rotation
    offset = 30
    dst_points = np.float32([
        [offset, offset],
        [width-1-offset, offset//2],
        [offset//2, height-1-offset],
        [width-1-offset//2, height-1-offset//2]
    ])
    
    # Apply perspective transformation
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    skewed = cv2.warpPerspective(image, M, (width, height))
    
    cv2.imwrite(output_path, skewed)


def create_anchor_test_image(output_path: str):
    """Create a test image with stable anchor patterns."""
    # Create white image
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Add some geometric patterns as stable anchors
    
    # Header text simulation (black rectangle)
    cv2.rectangle(image, (50, 50), (750, 100), (0, 0, 0), -1)
    cv2.putText(image, "DEMO FORM", (300, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Checkbox patterns
    checkbox_size = 30
    for i, y in enumerate([200, 300, 400]):
        x = 100
        cv2.rectangle(image, (x, y), (x + checkbox_size, y + checkbox_size), (0, 0, 0), 2)
        cv2.putText(image, f"Option {i+1}", (x + 50, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Logo simulation (circle)
    cv2.circle(image, (700, 150), 40, (0, 0, 0), 3)
    cv2.putText(image, "LOGO", (670, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Field lines
    for y in [250, 350, 450]:
        cv2.line(image, (200, y), (500, y), (0, 0, 0), 1)
    
    cv2.imwrite(output_path, image)


def cleanup_demo_files():
    """Clean up demo files."""
    demo_files = [
        "demo_fiducials.png",
        "demo_skewed.png", 
        "demo_aligned.png",
        "demo_anchors.png",
        "demo_template.json",
        "demo_anchor_template.json",
        "demo_template_refined.json",
        "demo_snap_preview.png"
    ]
    
    removed = 0
    for file in demo_files:
        if Path(file).exists():
            Path(file).unlink()
            removed += 1
    
    if removed > 0:
        print(f"Cleaned up {removed} demo files")


def main():
    """Run alignment demos."""
    print("Enhanced OCR Alignment System Demo")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "fiducials":
            demo_fiducial_alignment()
        elif command == "anchors":
            demo_anchor_alignment()
        elif command == "snap-to-ink":
            demo_snap_to_ink_with_alignment()
        elif command == "cleanup":
            cleanup_demo_files()
        elif command == "all":
            demo_fiducial_alignment()
            demo_anchor_alignment()
            demo_snap_to_ink_with_alignment()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: fiducials, anchors, snap-to-ink, cleanup, all")
    else:
        print("\nAvailable demos:")
        print("  python demo_alignment.py fiducials     - Fiducial marker alignment")
        print("  python demo_alignment.py anchors       - Stable anchor alignment")
        print("  python demo_alignment.py snap-to-ink   - Enhanced snap-to-ink processing")
        print("  python demo_alignment.py all           - Run all demos")
        print("  python demo_alignment.py cleanup       - Clean up demo files")
        print("\nNote: Requires opencv-contrib-python for full functionality")


if __name__ == "__main__":
    main()