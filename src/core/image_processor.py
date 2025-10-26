"""
Image processing utilities for OCR
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from PIL import Image
import json
import os

class ImageProcessor:
    """Handles all image processing operations"""
    
    def __init__(self):
        self.templates_dir = os.path.join(os.getcwd(), "data", "templates")
    
    def align_to_template(self, image: np.ndarray, template_config: Dict) -> np.ndarray:
        """Align input image to canonical template"""
        # Load canonical template
        canon_path = os.path.join(self.templates_dir, template_config["canonical_image"])
        canonical = cv2.imread(canon_path, cv2.IMREAD_COLOR)
        
        if canonical is None:
            raise FileNotFoundError(f"Canonical template not found: {canon_path}")
        
        # Perform alignment
        aligned = self._align_to_canonical(image, canonical)
        return aligned
    
    def extract_roi(self, image: np.ndarray, roi: Dict, template_config: Dict) -> np.ndarray:
        """Extract ROI patch from aligned image"""
        H, W = image.shape[:2]
        canvas_w = template_config.get("canvas_width", W)
        canvas_h = template_config.get("canvas_height", H)
        
        # Convert relative coordinates to pixels with adaptive padding
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        
        # Add adaptive padding based on image size and field type
        padding_factor = self._get_padding_factor(roi.get("type", "text"), canvas_w, canvas_h)
        
        # Convert to pixel coordinates
        if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
            # Relative coordinates
            px_x = int(x * W)
            px_y = int(y * H)
            px_w = int(w * W)
            px_h = int(h * H)
        else:
            # Already pixel coordinates
            px_x, px_y, px_w, px_h = int(x), int(y), int(w), int(h)
        
        # Apply adaptive padding
        px_x = max(0, px_x - padding_factor)
        px_y = max(0, px_y - padding_factor)
        px_w = min(W - px_x, px_w + 2 * padding_factor)
        px_h = min(H - px_y, px_h + 2 * padding_factor)
        
        # Extract and preprocess ROI
        roi_patch = image[px_y:px_y+px_h, px_x:px_x+px_w]
        roi_patch = self._preprocess_roi(roi_patch)
        
        return roi_patch
    
    def scale_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Scale image by given factor"""
        if scale == 1.0:
            return image
        
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    def _get_padding_factor(self, field_type: str, canvas_w: int, canvas_h: int) -> int:
        """Calculate adaptive padding based on field type and image size"""
        base_padding = min(canvas_w, canvas_h) * 0.005  # 0.5% of smaller dimension
        
        type_multipliers = {
            "text": 1.5,
            "digits": 1.0,
            "currency": 1.2,
            "date": 1.0
        }
        
        multiplier = type_multipliers.get(field_type, 1.0)
        return max(2, int(base_padding * multiplier))
    
    def _preprocess_roi(self, roi_patch: np.ndarray) -> np.ndarray:
        """Preprocess ROI patch for better OCR"""
        if len(roi_patch.shape) == 3:
            roi_patch = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2GRAY)
        
        # Apply deskewing and binarization
        roi_patch = self._deskew_and_binarize(roi_patch)
        
        return roi_patch
    
    def _align_to_canonical(self, input_img: np.ndarray, canonical: np.ndarray) -> np.ndarray:
        """
        Align input image to canonical template using feature matching
        """
        # Convert to grayscale
        if len(input_img.shape) == 3:
            input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        else:
            input_gray = input_img.copy()
        
        if len(canonical.shape) == 3:
            canon_gray = cv2.cvtColor(canonical, cv2.COLOR_BGR2GRAY)
        else:
            canon_gray = canonical.copy()
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create(nfeatures=5000)
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(input_gray, None)
        kp2, des2 = sift.detectAndCompute(canon_gray, None)
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            print("Warning: Insufficient features for alignment, returning original")
            return input_img
        
        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            print("Warning: Insufficient good matches for alignment")
            return input_img
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            print("Warning: Could not compute homography")
            return input_img
        
        # Check for excessive distortion
        if self._is_distortion_excessive(M):
            print("Warning: Excessive distortion detected, using original image")
            return input_img
        
        # Apply transformation
        h, w = canonical.shape[:2]
        aligned = cv2.warpPerspective(input_img, M, (w, h))
        
        return aligned
    
    def _is_distortion_excessive(self, homography: np.ndarray, threshold: float = 3.0) -> bool:
        """Check if homography indicates excessive distortion"""
        try:
            # Decompose homography to check for reasonable transformation
            corners = np.array([
                [0, 0, 1],
                [100, 0, 1],
                [100, 100, 1],
                [0, 100, 1]
            ]).T
            
            transformed = homography @ corners
            transformed = transformed[:2] / transformed[2]
            
            # Check if transformation is reasonable (not too skewed)
            widths = [
                np.linalg.norm(transformed[:, 1] - transformed[:, 0]),
                np.linalg.norm(transformed[:, 2] - transformed[:, 3])
            ]
            heights = [
                np.linalg.norm(transformed[:, 3] - transformed[:, 0]),
                np.linalg.norm(transformed[:, 2] - transformed[:, 1])
            ]
            
            width_ratio = max(widths) / min(widths) if min(widths) > 0 else float('inf')
            height_ratio = max(heights) / min(heights) if min(heights) > 0 else float('inf')
            
            return width_ratio > threshold or height_ratio > threshold
            
        except:
            return True
    
    def _deskew_and_binarize(self, image: np.ndarray) -> np.ndarray:
        """Apply deskewing and binarization"""
        # Simple deskewing using Hough transform
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None and len(lines) > 0:
            angles = []
            for rho, theta in lines[:10]:  # Use top 10 lines
                angle = theta - np.pi/2
                angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.01:  # Only correct if angle is significant
                    center = (image.shape[1]//2, image.shape[0]//2)
                    M = cv2.getRotationMatrix2D(center, np.degrees(median_angle), 1.0)
                    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # Adaptive binarization
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return binary