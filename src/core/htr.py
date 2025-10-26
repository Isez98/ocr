from typing import Tuple
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import unicodedata
import torch

from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# Lazy singletons
_processor = None
_model = None
_large_processor = None
_large_model = None


def get_device():
  """Force CPU processing to avoid ROCm/CUDA conflicts causing segfaults"""
  device = torch.device("cpu")
  print("Using CPU (forced to avoid GPU conflicts)")
  return device


def get_model(use_large=False):
  global _processor, _model, _large_processor, _large_model
  device = get_device()
  
  if use_large:
    if _large_processor is None:
      _large_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    if _large_model is None:
      _large_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
      if device.type == "cuda":
        _large_model = _large_model.to(device)
      _large_model.eval()
      # Enable memory efficient attention if available
      if hasattr(_large_model.config, 'use_cache'):
        _large_model.config.use_cache = False
    return _large_processor, _large_model
  else:
    if _processor is None:
      _processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    if _model is None:
      _model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
      if device.type == "cuda":
        _model = _model.to(device)
      _model.eval()
      # Enable memory efficient attention if available
      if hasattr(_model.config, 'use_cache'):
        _model.config.use_cache = False
    return _processor, _model




def enhance_image_for_htr(pil_img: Image.Image) -> Image.Image:
  """Enhanced preprocessing specifically for handwriting recognition"""
  
  # Convert to grayscale if needed
  if pil_img.mode != 'L':
    pil_img = pil_img.convert('L')
  
  # Enhance contrast
  enhancer = ImageEnhance.Contrast(pil_img)
  pil_img = enhancer.enhance(1.5)
  
  # Enhance sharpness
  enhancer = ImageEnhance.Sharpness(pil_img)
  pil_img = enhancer.enhance(1.2)
  
  # Apply slight gaussian blur to reduce noise
  pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))
  
  # Convert to numpy for OpenCV operations
  img_array = np.array(pil_img)
  
  # Adaptive histogram equalization (CLAHE)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  img_array = clahe.apply(img_array)
  
  # Advanced denoising
  img_array = cv2.fastNlMeansDenoising(img_array, h=10, templateWindowSize=7, searchWindowSize=21)
  
  # Improved binarization using Otsu's method with Gaussian blur
  blur = cv2.GaussianBlur(img_array, (5,5), 0)
  _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  
  # Morphological operations to clean up the image
  kernel = np.ones((2,2), np.uint8)
  binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
  binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
  
  # Convert back to PIL and ensure RGB for TrOCR
  enhanced_pil = Image.fromarray(binary).convert('RGB')
  
  return enhanced_pil


def deskew_and_binarize(img: np.ndarray, focus_handwriting: bool = False) -> np.ndarray:
  """Standard preprocessing with optional light handwriting enhancement"""
  
  if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  # Bilateral filter to reduce noise while preserving edges
  filtered = cv2.bilateralFilter(img, 9, 75, 75)
  
  if focus_handwriting:
    # Slightly more aggressive thresholding for handwriting
    binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
  else:
    # Standard adaptive threshold
    binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
  
  # Light morphological operations to clean up
  kernel = np.ones((2,2), np.uint8)
  binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
  binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
  
  return binary




def align_to_canonical(img: np.ndarray, canonical: np.ndarray) -> np.ndarray:
  """Improved alignment with better filtering and fallback"""
  
  # Try ORB with better parameters
  orb = cv2.ORB_create(5000, scaleFactor=1.2, nlevels=8)
  kp1, des1 = orb.detectAndCompute(img, None)
  kp2, des2 = orb.detectAndCompute(canonical, None)
  
  if des1 is None or des2 is None:
    print("No features detected, returning original image")
    return img
    
  # Use FLANN matcher for better matching
  FLANN_INDEX_LSH = 6
  index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
  search_params = dict(checks=50)
  flann = cv2.FlannBasedMatcher(index_params, search_params)
  
  try:
    matches = flann.knnMatch(des1, des2, k=2)
  except:
    # Fallback to BF matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    raw_matches = bf.match(des1, des2)
    matches = [[m] for m in raw_matches]
  
  # Apply Lowe's ratio test for better matches
  good_matches = []
  for match_pair in matches:
    if len(match_pair) == 2:
      m, n = match_pair
      if m.distance < 0.7 * n.distance:
        good_matches.append(m)
    elif len(match_pair) == 1:
      good_matches.append(match_pair[0])
  
  print(f"Found {len(good_matches)} good matches")
  
  if len(good_matches) < 20:
    print("Not enough good matches, returning original image")
    return img
  
  # Use only the best matches
  good_matches = sorted(good_matches, key=lambda x: x.distance)[:min(100, len(good_matches))]
  
  src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
  dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
  
  # Try homography with stricter RANSAC parameters
  H, mask = cv2.findHomography(src_pts, dst_pts, 
                               cv2.RANSAC, 
                               ransacReprojThreshold=3.0,
                               maxIters=5000,
                               confidence=0.995)
  
  if H is None:
    print("Homography estimation failed, returning original image")
    return img
  
  # Check if homography is reasonable (not too distorted)
  corners = np.float32([[0,0], [img.shape[1],0], [img.shape[1],img.shape[0]], [0,img.shape[0]]]).reshape(-1,1,2)
  transformed_corners = cv2.perspectiveTransform(corners, H)
  
  # Check for extreme distortion
  area_original = img.shape[0] * img.shape[1]
  area_transformed = cv2.contourArea(transformed_corners)
  area_ratio = area_transformed / area_original
  
  if area_ratio < 0.1 or area_ratio > 10:
    print(f"Extreme distortion detected (area ratio: {area_ratio:.3f}), returning original image")
    return img
  
  # Check inlier ratio
  if mask is not None:
    inlier_ratio = np.sum(mask) / len(mask)
    print(f"Inlier ratio: {inlier_ratio:.3f}")
    if inlier_ratio < 0.3:
      print("Low inlier ratio, returning original image")
      return img
  
  h, w = canonical.shape[:2]
  warped = cv2.warpPerspective(img, H, (w, h))
  print("Alignment successful")
  return warped




def crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
  h_img, w_img = img.shape[:2]
  x = max(0, min(x, w_img-1))
  y = max(0, min(y, h_img-1))
  w = max(1, min(w, w_img - x))
  h = max(1, min(h, h_img - y))
  return img[y:y+h, x:x+w]




def htr_read(pil_img: Image.Image, field_type: str = "text") -> Tuple[str, float]:
  """Hybrid HTR: GPU for model inference, CPU for image processing"""
  
  # Basic image preprocessing (CPU only to avoid ROCm conflicts)
  if pil_img.mode != 'RGB':
    pil_img = pil_img.convert('RGB')
  
  # Get model and processor
  processor, model = get_model(use_large=False)
  
  try:
    # Process image on CPU (preprocessor handles this)
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
    
    # Move to GPU only for model inference if available
    device = next(model.parameters()).device
    if device.type == "cuda":
      pixel_values = pixel_values.to(device)
    
    with torch.no_grad():
      # Generate with beam search for better results
      generated_ids = model.generate(
        pixel_values,
        max_length=50,
        num_beams=3,
        early_stopping=True,
        do_sample=False
      )
      
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    # Calculate basic confidence based on text characteristics
    confidence = calculate_confidence(text, field_type, pil_img.size)
    
    return text, confidence
    
  except Exception as e:
    print(f"HTR processing error: {e}")
    return "", 0.0


def calculate_confidence(text: str, field_type: str, image_size: tuple) -> float:
  """Calculate confidence score based on text characteristics and field type"""
  if not text:
    return 0.0
  
  base_confidence = 0.3  # Start with base confidence
  
  # Length-based confidence
  expected_lengths = {
    "digits": (1, 3),
    "currency": (3, 10),
    "date": (8, 12),
    "text": (2, 50)
  }
  
  min_len, max_len = expected_lengths.get(field_type, (1, 50))
  if min_len <= len(text) <= max_len:
    base_confidence += 0.2
  
  # Character pattern matching
  if field_type == "digits":
    if text.replace(" ", "").isdigit():
      base_confidence += 0.3
  elif field_type == "currency":
    import re
    if re.match(r'^\d+\.?\d*$', text.replace(" ", "").replace(",", "")):
      base_confidence += 0.3
  elif field_type == "date":
    import re
    if re.search(r'\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}|\d{4}-\d{2}-\d{2}', text):
      base_confidence += 0.3
  
  # Penalize special characters in simple fields
  if field_type in ["digits", "currency"]:
    special_chars = sum(1 for c in text if not c.isalnum() and c not in ".,/-$ ")
    base_confidence -= min(0.2, special_chars * 0.05)
  
  # Image size factor (larger patches generally yield better results)
  width, height = image_size
  if width > 100 and height > 30:
    base_confidence += 0.1
  
  return min(1.0, max(0.0, base_confidence))




def postprocess(text: str, ftype: str) -> str:
  """Enhanced postprocessing with better field-specific cleaning"""
  # Basic normalization
  text = unicodedata.normalize("NFC", text)
  text = text.strip()
  
  if ftype == "digits":
    import re
    # Extract only digits and basic formatting
    text = re.sub(r"[^0-9+\-\s]", "", text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # If it's just a number, clean it up
    numbers = re.findall(r'\d+', text)
    if numbers:
      text = numbers[0]  # Take the first number found
      
  elif ftype == "currency":
    import re
    # Remove template words that contaminate currency fields
    template_words = ['rate', 'night', 'cleaning', 'fee', 'deposit', 'security', 'exchange', 'total']
    for word in template_words:
        text = re.sub(rf'\b{word}\b', '', text, flags=re.IGNORECASE)
    
    # More sophisticated currency processing
    # First, try to extract number patterns
    text = text.replace('$', '').replace('€', '').replace('£', '')
    
    # Handle common OCR mistakes
    text = text.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
    
    # Extract digits, dots, and commas
    cleaned = re.sub(r"[^0-9.,]", "", text)
    
    # Handle different decimal formats
    if ',' in cleaned and '.' in cleaned:
      # Determine which is decimal separator
      comma_pos = cleaned.rfind(',')
      dot_pos = cleaned.rfind('.')
      if dot_pos > comma_pos:
        # Dot is decimal separator, comma is thousands
        cleaned = cleaned.replace(',', '')
      else:
        # Comma is decimal separator, dot is thousands
        cleaned = cleaned.replace('.', '').replace(',', '.')
    elif ',' in cleaned:
      # Could be thousands or decimal separator
      parts = cleaned.split(',')
      if len(parts) == 2 and len(parts[1]) <= 2:
        # Likely decimal separator
        cleaned = cleaned.replace(',', '.')
      else:
        # Likely thousands separator
        cleaned = cleaned.replace(',', '')
    
    try:
      v = float(cleaned)
      text = f"{v:.2f}"
    except (ValueError, TypeError):
      text = cleaned
      
  elif ftype == "date":
    import re
    # Remove template words that contaminate date fields
    template_words = ['check', 'in', 'out', 'date', 'time', 'today']
    for word in template_words:
        text = re.sub(rf'\b{word}\b', '', text, flags=re.IGNORECASE)
    
    # Enhanced date processing
    text = text.replace(" ", "").replace(".", "/").replace("-", "/")
    
    # Handle common OCR mistakes in dates
    text = text.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
    
    # Extract all numbers
    numbers = re.findall(r"(\d{1,4})", text)
    if len(numbers) >= 3:
      a, b, c = numbers[0], numbers[1], numbers[2]
      try:
        # Convert to integers to validate
        day, month, year = int(a), int(b), int(c)
        
        # Determine format based on number sizes
        if len(c) == 4:  # DD/MM/YYYY
          if 1 <= day <= 31 and 1 <= month <= 12:
            text = f"{c}-{month:02d}-{day:02d}"
        elif len(a) == 4:  # YYYY/MM/DD
          if 1 <= month <= 12 and 1 <= day <= 31:
            text = f"{a}-{month:02d}-{day:02d}"
        else:  # Assume DD/MM/YY and convert to 20YY
          if year < 50:
            year += 2000
          elif year < 100:
            year += 1900
          if 1 <= day <= 31 and 1 <= month <= 12:
            text = f"{year}-{month:02d}-{day:02d}"
      except (ValueError, TypeError):
        pass
        
  elif ftype == "text":
    # Enhanced text cleaning to remove template contamination
    import re
    
    # Remove excessive whitespace first
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Define common template words that often contaminate name fields
    template_noise = [
        'responsible', 'guest', 'name', 'names', 'phone', 'provided', 'id',
        'address', 'signature', 'email', 'today', 'date', 'time', 'development',
        'condominium', 'security', 'deposit', 'rate', 'cleaning', 'fee', 'nights',
        'adults', 'children', 'check', 'in', 'out', 'registered', 'by'
    ]
    
    # Remove template noise words (case insensitive)
    words = text.split()
    cleaned_words = []
    
    for word in words:
        # Clean up the word first
        clean_word = re.sub(r'[^\w\s\-\']', '', word.lower())
        
        # Skip if it's a template noise word
        if clean_word not in template_noise and len(clean_word) > 1:
            # Keep the original casing
            cleaned_words.append(word)
    
    text = ' '.join(cleaned_words)
    
    # Extract likely name patterns (capitalize words appropriately)
    if text:
        # Look for capitalized word patterns that are likely names
        name_pattern = re.findall(r'\b[A-Z][a-z]+\b', text)
        if name_pattern and len(name_pattern) >= 1:
            # Keep multiple name parts (first name, last name, etc.)
            # Take up to 3 name components (handles first, middle, last)
            text = ' '.join(name_pattern[:3])
        else:
            # If no proper capitalization, try to extract meaningful words
            words = text.split()
            name_words = []
            for word in words:
                clean = re.sub(r'[^\w\-\']', '', word)
                if len(clean) > 1 and clean.isalpha():
                    name_words.append(clean.capitalize())
                    if len(name_words) == 3:  # Max 3 name parts
                        break
            if name_words:
                text = ' '.join(name_words)
    
    # Remove remaining standalone special characters
    text = re.sub(r'\s[^\w\s]\s', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Final cleanup - remove quotes and other artifacts
    text = text.replace('"', '').replace("'", "").strip()
  
  return text