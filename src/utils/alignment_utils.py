#!/usr/bin/env python3
"""
Enhanced alignment utilities for OCR template creation.
Provides fiducial marker detection and stable anchor-based alignment for perfect warp.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FiducialMarkerDetector:
    """Detects ArUco fiducial markers for precise alignment."""
    
    def __init__(self, marker_dict=cv2.aruco.DICT_4X4_50):
        """
        Initialize ArUco detector.
        
        Args:
            marker_dict: ArUco dictionary to use (default: DICT_4X4_50)
        """
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(marker_dict)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
    
    def detect_markers(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[int], Optional[np.ndarray]]:
        """
        Detect ArUco markers in image.
        
        Args:
            image: Input image (grayscale or BGR)
            
        Returns:
            Tuple of (corners, ids, rejected_candidates)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        corners, ids, rejected = self.detector.detectMarkers(gray)
        return corners, ids, rejected
    
    def get_marker_centers(self, corners: List[np.ndarray], ids: List[int]) -> Dict[int, Tuple[float, float]]:
        """
        Calculate center points of detected markers.
        
        Args:
            corners: List of marker corner arrays
            ids: List of marker IDs
            
        Returns:
            Dictionary mapping marker ID to (x, y) center coordinate
        """
        centers = {}
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                # Calculate center as mean of corner points
                corner_points = corners[i][0]
                center_x = np.mean(corner_points[:, 0])
                center_y = np.mean(corner_points[:, 1])
                centers[marker_id] = (center_x, center_y)
        return centers


class StableAnchorDetector:
    """Detects stable anchors using logos, keywords, and geometric shapes."""
    
    def __init__(self):
        """Initialize anchor detector with default parameters."""
        self.min_anchors = 3
        
    def detect_logos(self, image: np.ndarray, template_logos: List[np.ndarray] = None) -> List[Tuple[int, int]]:
        """
        Detect printed logos using template matching.
        
        Args:
            image: Input image
            template_logos: List of logo templates to match against
            
        Returns:
            List of (x, y) logo center coordinates
        """
        if template_logos is None:
            return []
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        logo_centers = []
        
        for template in template_logos:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
            
            # Multi-scale template matching
            for scale in [0.8, 1.0, 1.2]:
                scaled_template = cv2.resize(template_gray, None, fx=scale, fy=scale)
                
                if scaled_template.shape[0] > gray.shape[0] or scaled_template.shape[1] > gray.shape[1]:
                    continue
                    
                result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= 0.7)  # High threshold for logos
                
                for pt in zip(*locations[::-1]):
                    center_x = pt[0] + scaled_template.shape[1] // 2
                    center_y = pt[1] + scaled_template.shape[0] // 2
                    logo_centers.append((center_x, center_y))
        
        # Remove duplicates (points too close together)
        return self._remove_duplicate_points(logo_centers, min_distance=50)
    
    def detect_text_anchors(self, image: np.ndarray, keywords: List[str] = None) -> List[Tuple[int, int]]:
        """
        Detect text-based anchors using OCR to find specific keywords.
        
        Args:
            image: Input image
            keywords: List of keywords to search for (e.g., ["FORM", "DATE", "NAME"])
            
        Returns:
            List of (x, y) text anchor coordinates
        """
        if keywords is None:
            keywords = ["FORM", "DATE", "NAME", "ADDRESS", "PHONE", "EMAIL"]
            
        # Use basic contour detection for now
        # In production, you might want to integrate with actual OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find text regions using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 2))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_anchors = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (likely text regions)
            if 50 < w < 300 and 15 < h < 50:
                center_x = x + w // 2
                center_y = y + h // 2
                text_anchors.append((center_x, center_y))
        
        return text_anchors[:10]  # Limit to top 10 candidates
    
    def detect_checkbox_corners(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect checkbox corners using Canny edge detection and Hough lines.
        
        Args:
            image: Input image
            
        Returns:
            List of (x, y) checkbox corner coordinates
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return []
        
        # Find line intersections (potential corners)
        corners = []
        for i, line1 in enumerate(lines):
            for line2 in lines[i+1:]:
                intersection = self._line_intersection(line1[0], line2[0])
                if intersection is not None:
                    x, y = intersection
                    if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                        corners.append((int(x), int(y)))
        
        # Filter corners that form rectangular patterns
        return self._filter_rectangular_corners(corners)
    
    def _line_intersection(self, line1, line2):
        """Calculate intersection of two lines in polar form."""
        rho1, theta1 = line1
        rho2, theta2 = line2
        
        # Convert to cartesian form
        A1, B1 = np.cos(theta1), np.sin(theta1)
        A2, B2 = np.cos(theta2), np.sin(theta2)
        
        det = A1 * B2 - A2 * B1
        if abs(det) < 1e-10:  # Lines are parallel
            return None
            
        x = (B2 * rho1 - B1 * rho2) / det
        y = (A1 * rho2 - A2 * rho1) / det
        
        return (x, y)
    
    def _filter_rectangular_corners(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Filter corners that likely belong to rectangular shapes (checkboxes)."""
        if len(corners) < 4:
            return corners
            
        # Simple filtering: remove points too close together
        return self._remove_duplicate_points(corners, min_distance=20)
    
    def _remove_duplicate_points(self, points: List[Tuple[int, int]], min_distance: int = 30) -> List[Tuple[int, int]]:
        """Remove duplicate points that are too close together."""
        if not points:
            return []
            
        filtered = [points[0]]
        for point in points[1:]:
            is_duplicate = False
            for existing in filtered:
                distance = np.sqrt((point[0] - existing[0])**2 + (point[1] - existing[1])**2)
                if distance < min_distance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(point)
        
        return filtered
    
    def detect_all_anchors(self, image: np.ndarray, 
                          template_logos: List[np.ndarray] = None,
                          keywords: List[str] = None) -> List[Tuple[int, int]]:
        """
        Detect all types of stable anchors.
        
        Args:
            image: Input image
            template_logos: Logo templates for matching
            keywords: Keywords to search for
            
        Returns:
            List of all detected anchor points
        """
        all_anchors = []
        
        # Detect logos
        logo_anchors = self.detect_logos(image, template_logos)
        all_anchors.extend(logo_anchors)
        logger.info(f"Detected {len(logo_anchors)} logo anchors")
        
        # Detect text anchors
        text_anchors = self.detect_text_anchors(image, keywords)
        all_anchors.extend(text_anchors)
        logger.info(f"Detected {len(text_anchors)} text anchors")
        
        # Detect checkbox corners
        checkbox_anchors = self.detect_checkbox_corners(image)
        all_anchors.extend(checkbox_anchors)
        logger.info(f"Detected {len(checkbox_anchors)} checkbox anchors")
        
        # Remove duplicates across all anchor types
        all_anchors = self._remove_duplicate_points(all_anchors, min_distance=30)
        
        logger.info(f"Total unique anchors detected: {len(all_anchors)}")
        return all_anchors


class AlignmentEngine:
    """Main alignment engine combining fiducial markers and stable anchors."""
    
    def __init__(self):
        """Initialize alignment engine."""
        self.fiducial_detector = FiducialMarkerDetector()
        self.anchor_detector = StableAnchorDetector()
        self.min_points_for_homography = 4
    
    def compute_alignment_homography(self, 
                                   runtime_image: np.ndarray,
                                   canonical_points: Dict[str, Tuple[float, float]],
                                   use_fiducials: bool = True,
                                   template_logos: List[np.ndarray] = None,
                                   keywords: List[str] = None) -> Optional[np.ndarray]:
        """
        Compute homography matrix for alignment.
        
        Args:
            runtime_image: Image to be aligned
            canonical_points: Reference points from template
            use_fiducials: Whether to try fiducial marker detection first
            template_logos: Logo templates for anchor detection
            keywords: Keywords for text anchor detection
            
        Returns:
            3x3 homography matrix or None if alignment failed
        """
        runtime_points = []
        canonical_coords = []
        
        # Try fiducial markers first if enabled
        if use_fiducials:
            homography = self._try_fiducial_alignment(runtime_image, canonical_points)
            if homography is not None:
                logger.info("Successfully aligned using fiducial markers")
                return homography
        
        # Fall back to stable anchor detection
        logger.info("Attempting alignment using stable anchors")
        anchors = self.anchor_detector.detect_all_anchors(
            runtime_image, template_logos, keywords
        )
        
        if len(anchors) < self.anchor_detector.min_anchors:
            logger.error(f"Insufficient anchors detected: {len(anchors)} < {self.anchor_detector.min_anchors}")
            return None
        
        # Match anchors to canonical points (simplified matching)
        matched_pairs = self._match_anchors_to_canonical(anchors, canonical_points)
        
        if len(matched_pairs) < self.min_points_for_homography:
            logger.error(f"Insufficient matched pairs: {len(matched_pairs)} < {self.min_points_for_homography}")
            return None
        
        # Extract points for homography calculation
        for runtime_pt, canonical_pt in matched_pairs:
            runtime_points.append(runtime_pt)
            canonical_coords.append(canonical_pt)
        
        # Compute homography
        runtime_points = np.array(runtime_points, dtype=np.float32)
        canonical_coords = np.array(canonical_coords, dtype=np.float32)
        
        homography, mask = cv2.findHomography(
            runtime_points, canonical_coords, 
            cv2.RANSAC, 5.0
        )
        
        if homography is not None:
            logger.info(f"Successfully computed homography using {len(matched_pairs)} anchor pairs")
        
        return homography
    
    def _try_fiducial_alignment(self, 
                               runtime_image: np.ndarray,
                               canonical_points: Dict[str, Tuple[float, float]]) -> Optional[np.ndarray]:
        """
        Attempt alignment using fiducial markers.
        
        Args:
            runtime_image: Image to align
            canonical_points: Should contain 'fiducials' key with marker data
            
        Returns:
            Homography matrix or None if failed
        """
        if 'fiducials' not in canonical_points:
            return None
        
        # Detect markers in runtime image
        corners, ids, _ = self.fiducial_detector.detect_markers(runtime_image)
        
        if ids is None or len(ids) < 4:
            logger.warning(f"Insufficient fiducial markers detected: {len(ids) if ids is not None else 0}")
            return None
        
        # Get marker centers
        runtime_centers = self.fiducial_detector.get_marker_centers(corners, ids)
        canonical_fiducials = canonical_points['fiducials']
        
        # Match detected markers to canonical positions
        runtime_points = []
        canonical_coords = []
        
        for marker_id, canonical_pos in canonical_fiducials.items():
            if marker_id in runtime_centers:
                runtime_points.append(runtime_centers[marker_id])
                canonical_coords.append(canonical_pos)
        
        if len(runtime_points) < 4:
            logger.warning(f"Insufficient matching fiducial pairs: {len(runtime_points)}")
            return None
        
        # Compute homography
        runtime_points = np.array(runtime_points, dtype=np.float32)
        canonical_coords = np.array(canonical_coords, dtype=np.float32)
        
        homography, _ = cv2.findHomography(runtime_points, canonical_coords)
        return homography
    
    def _match_anchors_to_canonical(self, 
                                   anchors: List[Tuple[int, int]],
                                   canonical_points: Dict[str, Tuple[float, float]]) -> List[Tuple[Tuple[int, int], Tuple[float, float]]]:
        """
        Match detected anchors to canonical reference points.
        This is a simplified matching strategy - in production you might want more sophisticated matching.
        
        Args:
            anchors: Detected anchor points
            canonical_points: Reference points from template
            
        Returns:
            List of matched (runtime_point, canonical_point) pairs
        """
        if 'anchors' not in canonical_points:
            # If no predefined anchors, try to match by spatial distribution
            return self._match_by_spatial_distribution(anchors, canonical_points)
        
        canonical_anchors = canonical_points['anchors']
        matched_pairs = []
        
        # Simple nearest-neighbor matching
        # In production, you'd want more robust feature-based matching
        used_anchors = set()
        
        for canonical_name, canonical_pos in canonical_anchors.items():
            best_match = None
            best_distance = float('inf')
            
            for i, anchor in enumerate(anchors):
                if i in used_anchors:
                    continue
                    
                distance = np.sqrt((anchor[0] - canonical_pos[0])**2 + (anchor[1] - canonical_pos[1])**2)
                if distance < best_distance:
                    best_distance = distance
                    best_match = i
            
            if best_match is not None and best_distance < 100:  # Reasonable threshold
                matched_pairs.append((anchors[best_match], canonical_pos))
                used_anchors.add(best_match)
        
        return matched_pairs
    
    def _match_by_spatial_distribution(self, 
                                     anchors: List[Tuple[int, int]],
                                     canonical_points: Dict[str, Tuple[float, float]]) -> List[Tuple[Tuple[int, int], Tuple[float, float]]]:
        """
        Match by spatial distribution when no predefined anchor mapping exists.
        Uses corner detection to match document corners.
        """
        if len(anchors) < 4:
            return []
        
        # Sort anchors by position to get corners
        anchors_sorted = sorted(anchors, key=lambda p: (p[1], p[0]))  # Sort by y, then x
        
        # Try to identify corners: top-left, top-right, bottom-left, bottom-right
        top_anchors = anchors_sorted[:len(anchors)//2]
        bottom_anchors = anchors_sorted[len(anchors)//2:]
        
        top_left = min(top_anchors, key=lambda p: p[0])
        top_right = max(top_anchors, key=lambda p: p[0])
        bottom_left = min(bottom_anchors, key=lambda p: p[0])
        bottom_right = max(bottom_anchors, key=lambda p: p[0])
        
        # Default canonical corners (you might want to make this configurable)
        canonical_corners = [
            (100, 100),    # top-left
            (1500, 100),   # top-right  
            (100, 2200),   # bottom-left
            (1500, 2200)   # bottom-right
        ]
        
        runtime_corners = [top_left, top_right, bottom_left, bottom_right]
        
        return list(zip(runtime_corners, canonical_corners))
    
    def apply_alignment(self, 
                       image: np.ndarray, 
                       homography: np.ndarray,
                       target_size: Tuple[int, int]) -> np.ndarray:
        """
        Apply homography transformation to align image.
        
        Args:
            image: Image to transform
            homography: 3x3 homography matrix
            target_size: (width, height) of output image
            
        Returns:
            Aligned image
        """
        return cv2.warpPerspective(image, homography, target_size)


def save_canonical_fiducials(canonical_image_path: str, 
                           template_path: str,
                           marker_ids: List[int] = [0, 1, 2, 3]) -> None:
    """
    Detect and save fiducial marker positions in canonical template.
    
    Args:
        canonical_image_path: Path to canonical image
        template_path: Path to template JSON file
        marker_ids: Expected marker IDs to look for
    """
    # Load canonical image
    image = cv2.imread(canonical_image_path)
    if image is None:
        raise ValueError(f"Could not load canonical image: {canonical_image_path}")
    
    # Detect markers
    detector = FiducialMarkerDetector()
    corners, ids, _ = detector.detect_markers(image)
    
    if ids is None or len(ids) < 4:
        raise ValueError(f"Could not detect sufficient fiducial markers. Found: {len(ids) if ids is not None else 0}")
    
    # Get marker centers
    centers = detector.get_marker_centers(corners, ids)
    
    # Load existing template
    with open(template_path, 'r') as f:
        template = json.load(f)
    
    # Add fiducial data
    template['fiducials'] = {int(marker_id): list(pos) for marker_id, pos in centers.items()}
    
    # Save updated template
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    logger.info(f"Saved {len(centers)} fiducial markers to template: {list(centers.keys())}")


def create_test_fiducial_image(output_path: str, 
                             image_size: Tuple[int, int] = (1653, 2339),
                             marker_size: int = 100) -> None:
    """
    Create a test image with ArUco markers at corners for testing.
    
    Args:
        output_path: Path to save test image
        image_size: (width, height) of image
        marker_size: Size of ArUco markers in pixels
    """
    # Create white image
    image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
    
    # Generate ArUco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # Marker positions (corners with some margin)
    margin = 50
    positions = [
        (margin, margin),  # top-left, marker ID 0
        (image_size[0] - margin - marker_size, margin),  # top-right, marker ID 1
        (margin, image_size[1] - margin - marker_size),  # bottom-left, marker ID 2
        (image_size[0] - margin - marker_size, image_size[1] - margin - marker_size)  # bottom-right, marker ID 3
    ]
    
    for i, (x, y) in enumerate(positions):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, i, marker_size)
        # Convert to 3-channel
        marker_img_3ch = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        image[y:y+marker_size, x:x+marker_size] = marker_img_3ch
    
    cv2.imwrite(output_path, image)
    logger.info(f"Created test fiducial image: {output_path}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python alignment_utils.py create_test <output_path>")
        print("  python alignment_utils.py save_fiducials <canonical_image> <template_json>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create_test":
        output_path = sys.argv[2] if len(sys.argv) > 2 else "test_fiducials.png"
        create_test_fiducial_image(output_path)
        
    elif command == "save_fiducials":
        if len(sys.argv) < 4:
            print("Usage: python alignment_utils.py save_fiducials <canonical_image> <template_json>")
            sys.exit(1)
        canonical_image = sys.argv[2]
        template_json = sys.argv[3]
        save_canonical_fiducials(canonical_image, template_json)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)