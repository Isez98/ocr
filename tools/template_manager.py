#!/usr/bin/env python3
"""
Template management utilities for enhanced OCR alignment.
Provides tools for adding fiducial markers to templates and managing anchor points.
"""

import cv2
import numpy as np
import json
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    from alignment_utils import (
        FiducialMarkerDetector, 
        StableAnchorDetector, 
        create_test_fiducial_image,
        save_canonical_fiducials
    )
except ImportError:
    print("Warning: alignment_utils.py not found. Some features will be unavailable.")
    FiducialMarkerDetector = None
    StableAnchorDetector = None
    create_test_fiducial_image = None
    save_canonical_fiducials = None


def add_fiducials_to_template(template_path: str, 
                            canonical_image_path: str,
                            output_template_path: Optional[str] = None) -> None:
    """
    Detect and add fiducial marker positions to an existing template.
    
    Args:
        template_path: Path to existing template JSON
        canonical_image_path: Path to canonical image with fiducial markers
        output_template_path: Path for output (defaults to overwriting input)
    """
    if save_canonical_fiducials is None:
        print("Error: alignment_utils.py required for fiducial detection")
        return
    
    if output_template_path is None:
        output_template_path = template_path
    
    # Create backup
    backup_path = template_path + '.backup'
    
    try:
        with open(template_path, 'r') as f:
            template = json.load(f)
        
        with open(backup_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"Created backup: {backup_path}")
        
        # Add fiducials to template
        save_canonical_fiducials(canonical_image_path, output_template_path)
        print(f"Successfully added fiducials to template: {output_template_path}")
        
    except Exception as e:
        print(f"Error adding fiducials: {e}")


def add_anchors_to_template(template_path: str,
                          canonical_image_path: str,
                          anchor_names: List[str] = None,
                          output_template_path: Optional[str] = None) -> None:
    """
    Detect and add stable anchor points to an existing template.
    
    Args:
        template_path: Path to existing template JSON
        canonical_image_path: Path to canonical image
        anchor_names: Names for detected anchors (optional)
        output_template_path: Path for output (defaults to overwriting input)
    """
    if StableAnchorDetector is None:
        print("Error: alignment_utils.py required for anchor detection")
        return
    
    if output_template_path is None:
        output_template_path = template_path
    
    # Load template and image
    with open(template_path, 'r') as f:
        template = json.load(f)
    
    image = cv2.imread(canonical_image_path)
    if image is None:
        print(f"Error: Could not load image {canonical_image_path}")
        return
    
    # Detect anchors
    detector = StableAnchorDetector()
    anchors = detector.detect_all_anchors(image)
    
    if len(anchors) < detector.min_anchors:
        print(f"Warning: Only {len(anchors)} anchors detected, minimum is {detector.min_anchors}")
    
    # Add anchor names
    if anchor_names is None:
        anchor_names = [f"anchor_{i}" for i in range(len(anchors))]
    
    # Create anchor dictionary
    anchor_dict = {}
    for i, (x, y) in enumerate(anchors):
        if i < len(anchor_names):
            anchor_dict[anchor_names[i]] = [float(x), float(y)]
    
    # Add to template
    template['anchors'] = anchor_dict
    
    # Save template
    with open(output_template_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Added {len(anchor_dict)} anchors to template: {output_template_path}")
    for name, pos in anchor_dict.items():
        print(f"  {name}: ({pos[0]:.1f}, {pos[1]:.1f})")


def create_template_with_fiducials(canonical_image_path: str,
                                 template_path: str,
                                 template_id: str = "form_with_fiducials") -> None:
    """
    Create a new template file with fiducial markers from a canonical image.
    
    Args:
        canonical_image_path: Path to canonical image with fiducial markers
        template_path: Path where to save the new template
        template_id: Identifier for the template
    """
    if FiducialMarkerDetector is None:
        print("Error: alignment_utils.py required for fiducial detection")
        return
    
    # Load image to get dimensions
    image = cv2.imread(canonical_image_path)
    if image is None:
        print(f"Error: Could not load image {canonical_image_path}")
        return
    
    height, width = image.shape[:2]
    
    # Create base template
    template = {
        "id": template_id,
        "canvas_width": width,
        "canvas_height": height,
        "canonical_image": canonical_image_path,
        "rois": []
    }
    
    # Detect and add fiducials
    detector = FiducialMarkerDetector()
    corners, ids, _ = detector.detect_markers(image)
    
    if ids is not None and len(ids) >= 4:
        centers = detector.get_marker_centers(corners, ids)
        template['fiducials'] = {int(marker_id): list(pos) for marker_id, pos in centers.items()}
        print(f"Detected {len(centers)} fiducial markers: {list(centers.keys())}")
    else:
        print("Warning: Could not detect sufficient fiducial markers")
    
    # Save template
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Created template with fiducials: {template_path}")


def visualize_template_alignment_points(template_path: str,
                                      canonical_image_path: str,
                                      output_path: str = None) -> None:
    """
    Create a visualization showing all alignment points in a template.
    
    Args:
        template_path: Path to template JSON
        canonical_image_path: Path to canonical image
        output_path: Path to save visualization (optional)
    """
    # Load template and image
    with open(template_path, 'r') as f:
        template = json.load(f)
    
    image = cv2.imread(canonical_image_path)
    if image is None:
        print(f"Error: Could not load image {canonical_image_path}")
        return
    
    vis_image = image.copy()
    
    # Draw fiducials if present
    if 'fiducials' in template:
        for marker_id, (x, y) in template['fiducials'].items():
            cv2.circle(vis_image, (int(x), int(y)), 20, (0, 255, 0), 3)
            cv2.putText(vis_image, f"F{marker_id}", (int(x-15), int(y-25)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw anchors if present
    if 'anchors' in template:
        for name, (x, y) in template['anchors'].items():
            cv2.circle(vis_image, (int(x), int(y)), 15, (255, 0, 0), 3)
            cv2.putText(vis_image, name, (int(x-20), int(y+30)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Draw ROIs
    for i, roi in enumerate(template.get('rois', [])):
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
        cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(vis_image, roi.get('name', f'ROI{i}'), (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Add legend
    legend_y = 30
    cv2.putText(vis_image, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_image, "Green circles: Fiducial markers", (10, legend_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(vis_image, "Blue circles: Stable anchors", (10, legend_y + 45), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(vis_image, "Red rectangles: ROIs", (10, legend_y + 65), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save or display
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved to: {output_path}")
    else:
        # Scale for display if too large
        height, width = vis_image.shape[:2]
        if height > 800:
            scale = 800 / height
            new_width = int(width * scale)
            vis_image = cv2.resize(vis_image, (new_width, 800))
        
        cv2.imshow('Template Alignment Points', vis_image)
        print("Press any key to close visualization...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_alignment_robustness(template_path: str,
                            test_images: List[str],
                            output_dir: str = "alignment_test_results") -> None:
    """
    Test alignment robustness across multiple test images.
    
    Args:
        template_path: Path to template JSON
        test_images: List of test image paths
        output_dir: Directory to save test results
    """
    try:
        from alignment_utils import AlignmentEngine
    except ImportError:
        print("Error: alignment_utils.py required for alignment testing")
        return
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load template
    with open(template_path, 'r') as f:
        template = json.load(f)
    
    # Initialize alignment engine
    engine = AlignmentEngine()
    
    results = []
    
    for i, image_path in enumerate(test_images):
        print(f"\nTesting image {i+1}/{len(test_images)}: {image_path}")
        
        # Load test image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Error: Could not load {image_path}")
            continue
        
        # Extract canonical points
        canonical_points = {}
        if 'fiducials' in template:
            canonical_points['fiducials'] = template['fiducials']
        if 'anchors' in template:
            canonical_points['anchors'] = template['anchors']
        
        # Test alignment
        homography = engine.compute_alignment_homography(image, canonical_points)
        
        result = {
            'image': image_path,
            'success': homography is not None,
            'fiducials_available': 'fiducials' in canonical_points,
            'anchors_available': 'anchors' in canonical_points
        }
        
        if homography is not None:
            # Apply alignment and save result
            target_size = (template['canvas_width'], template['canvas_height'])
            aligned = engine.apply_alignment(image, homography, target_size)
            
            output_path = Path(output_dir) / f"aligned_{i+1}.png"
            cv2.imwrite(str(output_path), aligned)
            
            result['output_path'] = str(output_path)
            print(f"  ✓ Alignment successful, saved to {output_path}")
        else:
            print(f"  ✗ Alignment failed")
        
        results.append(result)
    
    # Save test report
    report_path = Path(output_dir) / "alignment_test_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n--- Alignment Test Summary ---")
    print(f"Total images tested: {len(results)}")
    print(f"Successful alignments: {successful}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    print(f"Report saved to: {report_path}")


def main():
    """Main CLI interface for template management utilities."""
    parser = argparse.ArgumentParser(description="Template management utilities for enhanced OCR alignment")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add fiducials command
    add_fid_parser = subparsers.add_parser('add-fiducials', help='Add fiducial markers to template')
    add_fid_parser.add_argument('template', help='Path to template JSON file')
    add_fid_parser.add_argument('image', help='Path to canonical image with fiducials')
    add_fid_parser.add_argument('--output', help='Output template path (default: overwrite input)')
    
    # Add anchors command
    add_anchor_parser = subparsers.add_parser('add-anchors', help='Add stable anchors to template')
    add_anchor_parser.add_argument('template', help='Path to template JSON file')
    add_anchor_parser.add_argument('image', help='Path to canonical image')
    add_anchor_parser.add_argument('--output', help='Output template path (default: overwrite input)')
    add_anchor_parser.add_argument('--names', nargs='+', help='Names for detected anchors')
    
    # Create template command
    create_parser = subparsers.add_parser('create', help='Create new template with fiducials')
    create_parser.add_argument('image', help='Path to canonical image with fiducials')
    create_parser.add_argument('template', help='Path for new template JSON')
    create_parser.add_argument('--id', help='Template identifier', default='form_with_fiducials')
    
    # Visualize command
    vis_parser = subparsers.add_parser('visualize', help='Visualize template alignment points')
    vis_parser.add_argument('template', help='Path to template JSON file')
    vis_parser.add_argument('image', help='Path to canonical image')
    vis_parser.add_argument('--output', help='Path to save visualization image')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test alignment robustness')
    test_parser.add_argument('template', help='Path to template JSON file')
    test_parser.add_argument('images', nargs='+', help='Paths to test images')
    test_parser.add_argument('--output-dir', default='alignment_test_results', help='Output directory')
    
    # Create test fiducials command
    create_test_parser = subparsers.add_parser('create-test', help='Create test image with fiducial markers')
    create_test_parser.add_argument('output', help='Path for output test image')
    create_test_parser.add_argument('--width', type=int, default=1653, help='Image width')
    create_test_parser.add_argument('--height', type=int, default=2339, help='Image height')
    create_test_parser.add_argument('--marker-size', type=int, default=100, help='Marker size in pixels')
    
    args = parser.parse_args()
    
    if args.command == 'add-fiducials':
        add_fiducials_to_template(args.template, args.image, args.output)
        
    elif args.command == 'add-anchors':
        add_anchors_to_template(args.template, args.image, args.names, args.output)
        
    elif args.command == 'create':
        create_template_with_fiducials(args.image, args.template, args.id)
        
    elif args.command == 'visualize':
        visualize_template_alignment_points(args.template, args.image, args.output)
        
    elif args.command == 'test':
        test_alignment_robustness(args.template, args.images, args.output_dir)
        
    elif args.command == 'create-test':
        if create_test_fiducial_image is not None:
            create_test_fiducial_image(args.output, (args.width, args.height), args.marker_size)
        else:
            print("Error: alignment_utils.py required for creating test images")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()