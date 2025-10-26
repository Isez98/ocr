#!/usr/bin/env python3
"""
Snap to ink auto-refine functionality for OCR template creation.
Auto-tightens ROI boxes by detecting ink bounds and applying type-specific padding.
Now includes enhanced alignment with fiducial markers and stable anchors.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List
try:
    from alignment_utils import AlignmentEngine, FiducialMarkerDetector, StableAnchorDetector
except ImportError:
    # Fallback if alignment_utils is not available
    AlignmentEngine = None
    FiducialMarkerDetector = None
    StableAnchorDetector = None


def adaptive_threshold_and_morphology(roi_image: np.ndarray) -> np.ndarray:
    """
    Apply adaptive threshold and morphological opening to detect ink.
    
    Args:
        roi_image: Grayscale ROI image
        
    Returns:
        Binary image with ink detected
    """
    # Apply adaptive threshold to create binary image
    binary = cv2.adaptiveThreshold(
        roi_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Small morphology open (3x3) to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return cleaned


def find_ink_bounds(binary_image: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find tight bounds around ink using row/column projection sums.
    
    Args:
        binary_image: Binary image with ink as white pixels
        
    Returns:
        Tuple of (x_min, y_min, x_max, y_max) ink bounds
    """
    height, width = binary_image.shape
    
    # Build row projection (sum across columns for each row)
    row_sums = np.sum(binary_image, axis=1)
    # Build column projection (sum across rows for each column)
    col_sums = np.sum(binary_image, axis=0)
    
    # Find first and last non-zero rows
    non_zero_rows = np.where(row_sums > 0)[0]
    if len(non_zero_rows) == 0:
        # No ink detected, return original bounds
        return 0, 0, width, height
    
    y_min = non_zero_rows[0]
    y_max = non_zero_rows[-1]
    
    # Find first and last non-zero columns
    non_zero_cols = np.where(col_sums > 0)[0]
    if len(non_zero_cols) == 0:
        # No ink detected, return original bounds
        return 0, 0, width, height
    
    x_min = non_zero_cols[0]
    x_max = non_zero_cols[-1]
    
    return x_min, y_min, x_max + 1, y_max + 1  # +1 to make it inclusive


def get_type_specific_padding(roi_type: str, canvas_width: int, canvas_height: int) -> Dict[str, int]:
    """
    Get padding values based on field type and canvas size.
    
    Args:
        roi_type: Type of field (text, digits, currency, date, phone)
        canvas_width: Canvas width for relative padding calculation
        canvas_height: Canvas height for relative padding calculation
        
    Returns:
        Dictionary with padding values for top, bottom, left, right
    """
    # Base padding as 1% of canvas size
    base_pad_x = max(10, int(canvas_width * 0.01))
    base_pad_y = max(10, int(canvas_height * 0.01))
    
    # Clamp base padding to reasonable limits
    base_pad_x = min(base_pad_x, 20)
    base_pad_y = min(base_pad_y, 20)
    
    if roi_type in ["digits", "date", "phone"]:
        # Digits, dates, phones: +4–8 px vertical, +8–16 px horizontal
        return {
            "top": base_pad_y + 6,
            "bottom": base_pad_y + 6,
            "left": base_pad_x + 12,
            "right": base_pad_x + 12
        }
    elif roi_type == "currency":
        # Currency: extra right padding for decimals/symbols
        return {
            "top": base_pad_y + 6,
            "bottom": base_pad_y + 6,
            "left": base_pad_x + 8,
            "right": base_pad_x + 20  # Extra right padding
        }
    else:  # text (names/addresses)
        # Text: +10–20 px vertical for ascenders/accents
        return {
            "top": base_pad_y + 15,
            "bottom": base_pad_y + 15,
            "left": base_pad_x + 10,
            "right": base_pad_x + 10
        }


def align_image_to_template(image: np.ndarray, 
                           template: Dict,
                           use_fiducials: bool = True,
                           template_logos: List[np.ndarray] = None,
                           keywords: List[str] = None) -> Optional[np.ndarray]:
    """
    Align input image to template using fiducial markers or stable anchors.
    
    Args:
        image: Input image to align
        template: Template dictionary with canonical reference points
        use_fiducials: Whether to try fiducial marker alignment first
        template_logos: Logo templates for anchor detection
        keywords: Keywords for text anchor detection
        
    Returns:
        Aligned image or None if alignment failed
    """
    if AlignmentEngine is None:
        print("Warning: Enhanced alignment not available. Install alignment_utils.py")
        return image
    
    # Extract canonical points from template
    canonical_points = {}
    
    # Add fiducials if available
    if 'fiducials' in template:
        canonical_points['fiducials'] = template['fiducials']
    
    # Add anchor points if available
    if 'anchors' in template:
        canonical_points['anchors'] = template['anchors']
    
    # Initialize alignment engine
    engine = AlignmentEngine()
    
    # Compute homography
    homography = engine.compute_alignment_homography(
        image, canonical_points, use_fiducials, template_logos, keywords
    )
    
    if homography is None:
        print("Warning: Could not compute alignment homography")
        return None
    
    # Apply alignment
    target_size = (template['canvas_width'], template['canvas_height'])
    aligned_image = engine.apply_alignment(image, homography, target_size)
    
    return aligned_image


def validate_alignment_quality(original_image: np.ndarray, 
                             aligned_image: np.ndarray,
                             template: Dict) -> Dict[str, float]:
    """
    Validate the quality of image alignment by checking ROI content.
    
    Args:
        original_image: Original unaligned image
        aligned_image: Aligned image
        template: Template with ROI definitions
        
    Returns:
        Dictionary with quality metrics
    """
    metrics = {
        'ink_detection_score': 0.0,
        'edge_alignment_score': 0.0,
        'overall_quality': 0.0
    }
    
    if aligned_image is None:
        return metrics
    
    total_ink_original = 0
    total_ink_aligned = 0
    roi_count = 0
    
    for roi in template.get('rois', []):
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
        
        # Extract ROIs from both images
        if (x + w <= original_image.shape[1] and y + h <= original_image.shape[0] and
            x + w <= aligned_image.shape[1] and y + h <= aligned_image.shape[0]):
            
            original_roi = original_image[y:y+h, x:x+w]
            aligned_roi = aligned_image[y:y+h, x:x+w]
            
            # Convert to grayscale if needed
            if len(original_roi.shape) == 3:
                original_roi = cv2.cvtColor(original_roi, cv2.COLOR_BGR2GRAY)
            if len(aligned_roi.shape) == 3:
                aligned_roi = cv2.cvtColor(aligned_roi, cv2.COLOR_BGR2GRAY)
            
            # Detect ink in both ROIs
            orig_binary = adaptive_threshold_and_morphology(original_roi)
            aligned_binary = adaptive_threshold_and_morphology(aligned_roi)
            
            # Count ink pixels
            total_ink_original += np.sum(orig_binary > 0)
            total_ink_aligned += np.sum(aligned_binary > 0)
            roi_count += 1
    
    if roi_count > 0:
        # Higher ink detection in aligned image suggests better alignment
        if total_ink_original > 0:
            metrics['ink_detection_score'] = min(1.0, total_ink_aligned / total_ink_original)
        else:
            metrics['ink_detection_score'] = 1.0 if total_ink_aligned > 0 else 0.0
    
    # Simple edge alignment check
    if aligned_image is not None:
        gray_aligned = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY) if len(aligned_image.shape) == 3 else aligned_image
        edges = cv2.Canny(gray_aligned, 50, 150)
        
        # Check if edges are well-aligned (simplified metric)
        horizontal_edges = np.sum(edges, axis=1)
        vertical_edges = np.sum(edges, axis=0)
        
        # Look for strong horizontal/vertical patterns
        h_peaks = len([x for x in horizontal_edges if x > np.mean(horizontal_edges) + 2*np.std(horizontal_edges)])
        v_peaks = len([x for x in vertical_edges if x > np.mean(vertical_edges) + 2*np.std(vertical_edges)])
        
        metrics['edge_alignment_score'] = min(1.0, (h_peaks + v_peaks) / 20.0)
    
    # Overall quality is weighted average
    metrics['overall_quality'] = (
        0.7 * metrics['ink_detection_score'] + 
        0.3 * metrics['edge_alignment_score']
    )
    
    return metrics


def snap_roi_to_ink(
    original_image: np.ndarray, 
    roi: Dict, 
    canvas_width: int, 
    canvas_height: int
) -> Dict:
    """
    Auto-refine a single ROI by snapping to ink bounds with type-specific padding.
    
    Args:
        original_image: Full grayscale image
        roi: ROI dictionary with x, y, w, h, type fields
        canvas_width: Original canvas width
        canvas_height: Original canvas height
        
    Returns:
        Updated ROI dictionary with refined coordinates
    """
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    roi_type = roi.get("type", "text")
    
    # Extract ROI from original image
    roi_image = original_image[y:y+h, x:x+w]
    
    # Skip very small ROIs
    if roi_image.shape[0] < 10 or roi_image.shape[1] < 10:
        return roi
    
    # Apply adaptive threshold and morphology
    binary = adaptive_threshold_and_morphology(roi_image)
    
    # Find ink bounds within the ROI
    ink_x_min, ink_y_min, ink_x_max, ink_y_max = find_ink_bounds(binary)
    
    # If no ink detected, return original ROI
    if ink_x_min >= ink_x_max or ink_y_min >= ink_y_max:
        return roi
    
    # Get type-specific padding
    padding = get_type_specific_padding(roi_type, canvas_width, canvas_height)
    
    # Calculate new bounds with padding
    new_x = max(0, x + ink_x_min - padding["left"])
    new_y = max(0, y + ink_y_min - padding["top"])
    new_x_max = min(canvas_width, x + ink_x_max + padding["right"])
    new_y_max = min(canvas_height, y + ink_y_max + padding["bottom"])
    
    # Ensure minimum size
    new_w = max(20, new_x_max - new_x)
    new_h = max(10, new_y_max - new_y)
    
    # Update ROI
    refined_roi = roi.copy()
    refined_roi.update({
        "x": int(new_x),
        "y": int(new_y),
        "w": int(new_w),
        "h": int(new_h)
    })
    
    return refined_roi


def snap_template_to_ink(
    image_path: str, 
    template: Dict, 
    preview_mode: bool = False,
    enable_alignment: bool = False,  # Changed default to False
    use_fiducials: bool = True,
    template_logos: List[np.ndarray] = None,
    keywords: List[str] = None
) -> Tuple[Dict, Optional[np.ndarray]]:
    """
    Auto-refine all ROIs in a template by snapping to ink bounds.
    Now includes optional image alignment for perfect warp.
    
    Args:
        image_path: Path to the image to process
        template: Template dictionary with ROIs
        preview_mode: If True, return preview image with before/after boxes
        enable_alignment: Whether to attempt image alignment first
        use_fiducials: Whether to try fiducial marker alignment
        template_logos: Logo templates for anchor detection
        keywords: Keywords for text anchor detection
        
    Returns:
        Tuple of (refined_template, preview_image)
    """
    # Load image
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale for processing
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Attempt alignment if enabled
    aligned_image = None
    alignment_quality = None
    
    if enable_alignment:
        print("Attempting image alignment...")
        aligned_image = align_image_to_template(
            original_image, template, use_fiducials, template_logos, keywords
        )
        
        if aligned_image is not None:
            # Validate alignment quality
            alignment_quality = validate_alignment_quality(
                original_image, aligned_image, template
            )
            
            print(f"Alignment quality: {alignment_quality['overall_quality']:.2f}")
            print(f"  - Ink detection score: {alignment_quality['ink_detection_score']:.2f}")
            print(f"  - Edge alignment score: {alignment_quality['edge_alignment_score']:.2f}")
            
            # Use aligned image if quality is good enough
            if alignment_quality['overall_quality'] > 0.5:
                print("Using aligned image for ROI refinement")
                image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
            else:
                print("Alignment quality insufficient, using original image")
                aligned_image = None
        else:
            print("Alignment failed, using original image")
    
    canvas_width = template["canvas_width"]
    canvas_height = template["canvas_height"]
    
    # Create refined template
    refined_template = template.copy()
    refined_rois = []
    
    preview_image = None
    if preview_mode:
        preview_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    for i, roi in enumerate(template["rois"]):
        # Refine ROI
        refined_roi = snap_roi_to_ink(image, roi, canvas_width, canvas_height)
        refined_rois.append(refined_roi)
        
        # Add to preview if requested
        if preview_mode:
            # Draw original ROI in red
            cv2.rectangle(
                preview_image, 
                (roi["x"], roi["y"]), 
                (roi["x"] + roi["w"], roi["y"] + roi["h"]), 
                (0, 0, 255), 2
            )
            
            # Draw refined ROI in green
            cv2.rectangle(
                preview_image, 
                (refined_roi["x"], refined_roi["y"]), 
                (refined_roi["x"] + refined_roi["w"], refined_roi["y"] + refined_roi["h"]), 
                (0, 255, 0), 2
            )
            
            # Add label
            label = f"{i+1}. {roi['name']}"
            cv2.putText(
                preview_image, label, 
                (refined_roi["x"], refined_roi["y"] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
    
    refined_template["rois"] = refined_rois
    
    return refined_template, preview_image


def apply_snap_to_ink_interactive(image_path: str, 
                                template_path: str,
                                enable_alignment: bool = True,
                                use_fiducials: bool = True) -> None:
    """
    Interactive tool to preview and apply snap-to-ink refinement with alignment.
    
    Args:
        image_path: Path to image to process
        template_path: Path to template JSON file
        enable_alignment: Whether to attempt image alignment
        use_fiducials: Whether to prefer fiducial marker alignment
    """
    import json
    
    # Load template
    with open(template_path, 'r') as f:
        template = json.load(f)
    
    print(f"Loaded template '{template['id']}' with {len(template['rois'])} ROIs")
    
    # Generate preview
    try:
        refined_template, preview_image = snap_template_to_ink(
            image_path, template, preview_mode=True,
            enable_alignment=enable_alignment, use_fiducials=use_fiducials
        )
        
        # Show preview
        height, width = preview_image.shape[:2]
        if height > 800:
            scale = 800 / height
            new_width = int(width * scale)
            preview_image = cv2.resize(preview_image, (new_width, 800))
        
        cv2.imshow('Snap to Ink Preview (Red=Original, Green=Refined)', preview_image)
        
        print("\nPreview generated. ROI changes:")
        for i, (original, refined) in enumerate(zip(template["rois"], refined_template["rois"])):
            orig_coords = f"({original['x']}, {original['y']}, {original['w']}, {original['h']})"
            new_coords = f"({refined['x']}, {refined['y']}, {refined['w']}, {refined['h']})"
            
            if orig_coords != new_coords:
                print(f"  {i+1}. {original['name']}: {orig_coords} → {new_coords}")
            else:
                print(f"  {i+1}. {original['name']}: No change")
        
        print("\nPress 'y' to apply changes, 'n' to cancel, or any other key to exit")
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        
        if key == ord('y'):
            # Save refined template
            backup_path = template_path + '.backup'
            with open(backup_path, 'w') as f:
                json.dump(template, f, indent=2)
            print(f"Original template backed up to: {backup_path}")
            
            with open(template_path, 'w') as f:
                json.dump(refined_template, f, indent=2)
            print(f"Refined template saved to: {template_path}")
        else:
            print("Changes not applied")
            
    except Exception as e:
        print(f"Error processing template: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python snap_to_ink.py <image_path> <template_path> [--no-alignment] [--no-fiducials]")
        print("Example: python snap_to_ink.py sample_form.png templates/form_v1.json")
        print("Options:")
        print("  --no-alignment    Disable image alignment")
        print("  --no-fiducials    Disable fiducial marker detection")
        sys.exit(1)
    
    image_path = sys.argv[1]
    template_path = sys.argv[2]
    
    # Parse options
    enable_alignment = '--no-alignment' not in sys.argv
    use_fiducials = '--no-fiducials' not in sys.argv
    
    if enable_alignment:
        print("Enhanced alignment enabled")
        if use_fiducials:
            print("  - Fiducial marker detection: ON")
        else:
            print("  - Fiducial marker detection: OFF")
        print("  - Stable anchor detection: ON")
    else:
        print("Enhanced alignment disabled")
    
    apply_snap_to_ink_interactive(image_path, template_path, enable_alignment, use_fiducials)