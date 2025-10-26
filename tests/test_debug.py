#!/usr/bin/env python3

import io, json, os
import numpy as np
import cv2
from PIL import Image
from htr import align_to_canonical, deskew_and_binarize, crop, htr_read, postprocess

# Test the processing pipeline directly
def test_process():
    print("Starting test...")
    
    TEMPLATES_DIR = os.path.join(os.getcwd(), "templates")
    template_id = "form_v1"
    sample_file = "../samples/scan_form_1.jpg"
    
    # Load template
    tpath = os.path.join(TEMPLATES_DIR, f"{template_id}.json")
    print(f"Loading template from: {tpath}")
    
    if not os.path.exists(tpath):
        print("Template not found!")
        return
    
    cfg = json.load(open(tpath, "r", encoding="utf-8"))
    canon_path = os.path.join(TEMPLATES_DIR, os.path.basename(cfg["canonical_image"]))
    print(f"Loading canonical image from: {canon_path}")
    
    if not os.path.exists(canon_path):
        print("Canonical image not found!")
        return
    
    # Load sample image
    print(f"Loading sample image from: {sample_file}")
    if not os.path.exists(sample_file):
        print("Sample image not found!")
        return
    
    # Read image
    img = cv2.imread(sample_file, cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to load sample image with cv2, trying PIL...")
        pil = Image.open(sample_file).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    
    print(f"Image loaded successfully, shape: {img.shape}")
    
    # Load canonical
    canonical = cv2.imread(canon_path, cv2.IMREAD_COLOR)
    if canonical is None:
        print("Failed to load canonical image!")
        return
    
    print(f"Canonical loaded successfully, shape: {canonical.shape}")
    
    # Test alignment
    print("Testing alignment...")
    try:
        aligned = align_to_canonical(img, canonical)
        print(f"Alignment successful, shape: {aligned.shape}")
    except Exception as e:
        print(f"Alignment failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test gray conversion
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    print(f"Gray conversion successful, shape: {gray.shape}")
    
    # Test first ROI
    roi = cfg["rois"][0]
    print(f"Testing first ROI: {roi['name']}")
    
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    patch = crop(gray, x, y, w, h)
    print(f"Crop successful, patch shape: {patch.shape}")
    
    if patch.shape[0] < 10 or patch.shape[1] < 10:
        print("Patch too small, skipping...")
        return
    
    # Test binarization
    print("Testing binarization...")
    try:
        binimg = deskew_and_binarize(patch)
        print(f"Binarization successful, shape: {binimg.shape}")
    except Exception as e:
        print(f"Binarization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test RGB conversion
    rgb_img = cv2.cvtColor(binimg, cv2.COLOR_GRAY2RGB)
    pil = Image.fromarray(rgb_img)
    print(f"RGB conversion successful, PIL mode: {pil.mode}, size: {pil.size}")
    
    # Test HTR
    print("Testing HTR...")
    try:
        text, conf = htr_read(pil, roi.get("type", "text"))
        print(f"HTR successful: '{text}' (confidence: {conf})")
    except Exception as e:
        print(f"HTR failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test postprocess
    print("Testing postprocess...")
    try:
        processed_text = postprocess(text, roi.get("type", "text"))
        print(f"Postprocess successful: '{processed_text}'")
    except Exception as e:
        print(f"Postprocess failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("All tests passed!")

if __name__ == "__main__":
    test_process()