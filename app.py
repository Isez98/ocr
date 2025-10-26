import io, json, os, base64
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
from htr import align_to_canonical, deskew_and_binarize, crop, htr_read, postprocess
from snap_to_ink import snap_template_to_ink, snap_roi_to_ink
from active_learning import ActiveLearningPipeline

app = FastAPI()

# Initialize active learning pipeline
learning_pipeline = ActiveLearningPipeline()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

TEMPLATES_DIR = os.path.join(os.getcwd(), "templates")

def subtract_template_text(input_patch: np.ndarray, template_patch: np.ndarray) -> np.ndarray:
    """
    Remove printed template text to isolate handwritten content.
    
    Args:
        input_patch: Grayscale patch from input image
        template_patch: Corresponding grayscale patch from canonical template
        
    Returns:
        Cleaned patch with template text removed
    """
    # Ensure both patches are the same size
    if input_patch.shape != template_patch.shape:
        template_patch = cv2.resize(template_patch, (input_patch.shape[1], input_patch.shape[0]))
    
    # Convert to same data type
    input_patch = input_patch.astype(np.float32)
    template_patch = template_patch.astype(np.float32)
    
    # Normalize both images
    input_norm = cv2.normalize(input_patch, None, 0, 255, cv2.NORM_MINMAX)
    template_norm = cv2.normalize(template_patch, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply Gaussian blur to reduce noise
    input_blur = cv2.GaussianBlur(input_norm, (3, 3), 0)
    template_blur = cv2.GaussianBlur(template_norm, (3, 3), 0)
    
    # Calculate difference - handwriting should be darker than template
    diff = cv2.absdiff(input_blur, template_blur)
    
    # Threshold to isolate significant differences (handwriting)
    _, binary_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Create mask where handwriting is likely present
    # Areas with significant darkness in input but not in template
    input_dark = input_blur < 200  # Dark areas in input
    template_light = template_blur > 180  # Light areas in template
    handwriting_mask = np.logical_and(input_dark, template_light)
    
    # Combine difference-based and darkness-based detection
    final_mask = np.logical_or(binary_diff > 0, handwriting_mask)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((2, 2), np.uint8)
    final_mask = cv2.morphologyEx(final_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    # Apply mask to original input
    result = np.where(final_mask > 0, input_patch, 255)  # Keep handwriting, make rest white
    
    return result.astype(np.uint8)

@app.get("/health")
async def health():
  return {"ok": True}

@app.post("/debug-rois")
async def debug_rois(template_id: str = Form(...), file: UploadFile = File(...)):
  """Debug endpoint to visualize ROI extractions without OCR processing"""
  tpath = os.path.join(TEMPLATES_DIR, f"{template_id}.json")
  if not os.path.exists(tpath):
    return JSONResponse({"error": "unknown template"}, status_code=400)

  cfg = json.load(open(tpath, "r", encoding="utf-8"))
  canon_path = os.path.join(TEMPLATES_DIR, os.path.basename(cfg["canonical_image"]))
  if not os.path.exists(canon_path):
    return JSONResponse({"error": "missing canonical image"}, status_code=500)

  # Load and process image
  data = await file.read()
  file_bytes = np.frombuffer(data, dtype=np.uint8)
  img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
  if img is None:
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

  canonical = cv2.imread(canon_path, cv2.IMREAD_COLOR)
  if canonical is None:
    return JSONResponse({"error": "canonical not loadable"}, status_code=500)

  # Align image
  aligned = align_to_canonical(img, canonical)
  gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

  H, W = aligned.shape[:2]

  def rel_to_px(r):
      x, y, w, h = r["x"], r["y"], r["w"], r["h"]
      # if looks like relative, convert
      if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
          x, y, w, h = round(x*W), round(y*H), round(w*W), round(h*H)
      
      # Apply adaptive expansion for camera-captured forms
      base_pad = r.get("pad", 10)
      # Increase padding based on image size and type
      adaptive_pad = max(base_pad, int(min(W, H) * 0.005))  # 0.5% of smaller dimension
      
      # Type-specific expansion
      field_type = r.get("type", "text")
      if field_type == "text":
          adaptive_pad = int(adaptive_pad * 1.5)  # Text needs more vertical space
      elif field_type == "date":
          adaptive_pad = int(adaptive_pad * 1.2)  # Dates need horizontal space
      
      # Expand ROI with adaptive padding
      x = max(0, x - adaptive_pad)
      y = max(0, y - adaptive_pad)
      w = min(W - x, w + 2 * adaptive_pad)
      h = min(H - y, h + 2 * adaptive_pad)
      
      r = {**r, "x": x, "y": y, "w": w, "h": h}
      return r

  rois_px = [rel_to_px(r) for r in cfg["rois"]]

  # quick debug overlay
  def debug_overlay(img_bgr, rois, path="/tmp/roi_overlay.png"):
      v = img_bgr.copy()
      for r in rois:
          x,y,w,h = r["x"], r["y"], r["w"], r["h"]
          cv2.rectangle(v, (x,y), (x+w,y+h), (36,255,12), 2)
          cv2.putText(v, r["name"], (x, max(12, y-6)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
      cv2.imwrite(path, v)

  debug_overlay(aligned, rois_px)

  results = []
  for roi in rois_px:
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    patch = crop(gray, x, y, w, h)
    
    # Convert patch to base64 for visualization
    _, buffer = cv2.imencode('.png', patch)
    patch_b64 = base64.b64encode(buffer).decode('utf-8')
    
    results.append({
      "name": roi["name"],
      "type": roi["type"],
      "coordinates": {"x": x, "y": y, "w": w, "h": h},
      "patch_size": {"height": patch.shape[0], "width": patch.shape[1]},
      "patch_image_b64": patch_b64
    })

  return {
    "template_id": cfg["id"],
    "canvas_size": {"width": cfg["canvas_width"], "height": cfg["canvas_height"]},
    "aligned_image_size": {"height": gray.shape[0], "width": gray.shape[1]},
    "rois": results
  }

@app.post("/process")
async def process(template_id: str = Form(...), file: UploadFile = File(...)):
 try:
  print(f"Processing request with template_id: {template_id}, file: {file.filename}")
  
  tpath = os.path.join(TEMPLATES_DIR, f"{template_id}.json")
  if not os.path.exists(tpath):
   return JSONResponse({"error": "unknown template"}, status_code=400)

  cfg = json.load(open(tpath, "r", encoding="utf-8"))
  canon_path = os.path.join(TEMPLATES_DIR, os.path.basename(cfg["canonical_image"]))
  if not os.path.exists(canon_path):
   return JSONResponse({"error": "missing canonical image"}, status_code=500)

  # Load images
  print("Reading uploaded file...")
  data = await file.read()
  print(f"File size: {len(data)} bytes")
  
  file_bytes = np.frombuffer(data, dtype=np.uint8)
  img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
  if img is None:
   print("cv2.imdecode failed, trying PIL fallback...")
   # try PIL fallback
   pil = Image.open(io.BytesIO(data)).convert("RGB")
   img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
  
  print(f"Image loaded successfully, shape: {img.shape}")

  # Debug: save input image for inspection
  cv2.imwrite("/tmp/input_debug.png", img)

  canonical = cv2.imread(canon_path, cv2.IMREAD_COLOR)
  if canonical is None:
    return JSONResponse({"error": "canonical not loadable"}, status_code=500)

  # Debug: save canonical for inspection
  cv2.imwrite("/tmp/canonical_debug.png", canonical)

  print("Starting alignment...")
  print(f"Input image shape: {img.shape}")
  print(f"Canonical image shape: {canonical.shape}")
  # Align
  aligned = align_to_canonical(img, canonical)
  print(f"Aligned image shape: {aligned.shape}")
  print("Alignment completed")

  # Prepare gray for binarization
  gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

  H, W = aligned.shape[:2]

  def rel_to_px(r):
      x, y, w, h = r["x"], r["y"], r["w"], r["h"]
      # if looks like relative, convert
      if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
          x, y, w, h = round(x*W), round(y*H), round(w*W), round(h*H)
      r = {**r, "x": x, "y": y, "w": w, "h": h}
      return r

  rois_px = [rel_to_px(r) for r in cfg["rois"]]

  # quick debug overlay
  def debug_overlay(img_bgr, rois, path="/tmp/roi_overlay.png"):
      v = img_bgr.copy()
      for r in rois:
          x,y,w,h = r["x"], r["y"], r["w"], r["h"]
          cv2.rectangle(v, (x,y), (x+w,y+h), (36,255,12), 2)
          cv2.putText(v, r["name"], (x, max(12, y-6)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
      cv2.imwrite(path, v)

  debug_overlay(aligned, rois_px)

  results = []
  for i, roi in enumerate(rois_px):
   print(f"Processing ROI {i+1}/{len(rois_px)}: {roi['name']}")
   x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
   ftype = roi.get("type", "text")
   
   # Multi-scale approach: try original size, then slightly larger if first attempt fails
   scales = [1.0, 1.2, 1.4]  # Original, 20% larger, 40% larger
   best_result = None
   best_confidence = 0.0
   
   for scale in scales:
     # Apply scale to dimensions
     scaled_w = int(w * scale)
     scaled_h = int(h * scale)
     scaled_x = max(0, x - int((scaled_w - w) / 2))
     scaled_y = max(0, y - int((scaled_h - h) / 2))
     
     # Ensure we don't go out of bounds
     scaled_w = min(scaled_w, gray.shape[1] - scaled_x)
     scaled_h = min(scaled_h, gray.shape[0] - scaled_y)
     
     patch = crop(gray, scaled_x, scaled_y, scaled_w, scaled_h)
     
     # Skip patches that are too small for TrOCR processing
     if patch.shape[0] < 10 or patch.shape[1] < 10:
       continue
       
     binimg = deskew_and_binarize(patch, focus_handwriting=False)
     
     # Convert grayscale to RGB for TrOCR processor
     rgb_img = cv2.cvtColor(binimg, cv2.COLOR_GRAY2RGB)
     pil = Image.fromarray(rgb_img)
     text, conf = htr_read(pil, ftype)  # Pass field type for better processing
     text = postprocess(text, ftype)
     
     # Keep the result with highest confidence, or use original scale if confidence is good
     if conf > best_confidence or (scale == 1.0 and conf > 0.5):
       best_result = {
         "name": roi["name"],
         "text": text,
         "confidence": round(float(conf), 3),
         "scale_used": scale
       }
       best_confidence = conf
       
       # If we got good confidence at original scale, no need to try larger scales
       if scale == 1.0 and conf > 0.7:
         break
   
   if best_result:
     # Flag low confidence results for potential review
     if best_result["confidence"] < 0.8:
       best_result["needs_review"] = True
       best_result["review_reason"] = f"Low confidence ({best_result['confidence']:.2f})"
       print(f"⚠️  Low confidence for {best_result['name']}: '{best_result['text']}' (confidence: {best_result['confidence']:.2f})")
     else:
       best_result["needs_review"] = False
       
     results.append(best_result)
   else:
     # Fallback if all scales failed
     fallback_result = {
       "name": roi["name"],
       "text": "",
       "confidence": 0.0,
       "scale_used": 1.0,
       "needs_review": True,
       "review_reason": "No text detected"
     }
     results.append(fallback_result)
     print(f"⚠️  No text detected for {roi['name']}")

  # Summary of results needing review
  review_needed = [r for r in results if r.get("needs_review", False)]
  if review_needed:
    print(f"\n📋 {len(review_needed)} field(s) may need manual review:")
    for r in review_needed:
      print(f"   • {r['name']}: {r.get('review_reason', 'Unknown issue')}")

  print("Processing completed successfully")
  return {
   "template_id": cfg["id"],
   "fields": results,
   "review_summary": {
     "total_fields": len(results),
     "fields_needing_review": len(review_needed),
     "review_fields": [{"name": r["name"], "reason": r.get("review_reason")} for r in review_needed]
   }
  }
  
 except Exception as e:
  print(f"Error processing request: {str(e)}")
  import traceback
  traceback.print_exc()
  return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/snap-to-ink")
async def snap_to_ink_endpoint(template_id: str = Form(...)):
  """Apply snap-to-ink auto-refinement to an existing template"""
  tpath = os.path.join(TEMPLATES_DIR, f"{template_id}.json")
  if not os.path.exists(tpath):
    return JSONResponse({"error": "unknown template"}, status_code=400)

  cfg = json.load(open(tpath, "r", encoding="utf-8"))
  canon_path = os.path.join(TEMPLATES_DIR, os.path.basename(cfg["canonical_image"]))
  if not os.path.exists(canon_path):
    return JSONResponse({"error": "missing canonical image"}, status_code=500)

  try:
    # Load the canonical image to get dimensions for coordinate conversion
    canonical = cv2.imread(canon_path, cv2.IMREAD_COLOR)
    if canonical is None:
      return JSONResponse({"error": "canonical not loadable"}, status_code=500)
    
    H, W = canonical.shape[:2]
    
    # Convert relative coordinates to pixel coordinates for snap-to-ink
    def rel_to_px(r):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        # if looks like relative, convert
        if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
            x, y, w, h = round(x*W), round(y*H), round(w*W), round(h*H)
        r = {**r, "x": x, "y": y, "w": w, "h": h}
        return r
    
    # Create a pixel-coordinate version of the template
    cfg_px = cfg.copy()
    cfg_px["rois"] = [rel_to_px(r) for r in cfg["rois"]]
    
    # Apply snap-to-ink refinement
    refined_template, preview_image = snap_template_to_ink(
      canon_path, cfg_px, preview_mode=True
    )
    
    # Convert refined pixel coordinates back to relative coordinates
    def px_to_rel(r):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        x_rel, y_rel = x / W, y / H
        w_rel, h_rel = w / W, h / H
        r = {**r, "x": x_rel, "y": y_rel, "w": w_rel, "h": h_rel}
        return r
    
    # Convert refined ROIs back to relative coordinates
    refined_template["rois"] = [px_to_rel(r) for r in refined_template["rois"]]
    
    # Create backup of original
    backup_path = tpath + '.backup'
    with open(backup_path, 'w') as f:
      json.dump(cfg, f, indent=2)
    
    # Save refined template
    with open(tpath, 'w') as f:
      json.dump(refined_template, f, indent=2)
    
    # Convert preview to base64
    _, buffer = cv2.imencode('.png', preview_image)
    preview_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # Calculate changes summary
    changes = []
    for original, refined in zip(cfg["rois"], refined_template["rois"]):
      if (original['x'] != refined['x'] or original['y'] != refined['y'] or
          original['w'] != refined['w'] or original['h'] != refined['h']):
        changes.append({
          "name": original['name'],
          "original": {"x": original['x'], "y": original['y'], "w": original['w'], "h": original['h']},
          "refined": {"x": refined['x'], "y": refined['y'], "w": refined['w'], "h": refined['h']}
        })
    
    return {
      "template_id": cfg["id"],
      "backup_created": backup_path,
      "total_rois": len(cfg["rois"]),
      "rois_changed": len(changes),
      "changes": changes,
      "preview_image_b64": preview_b64
    }
    
  except Exception as e:
    return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/analyze-roi-coverage")
async def analyze_roi_coverage(template_id: str = Form(...), files: list[UploadFile] = File(...)):
  """Analyze ROI coverage across multiple samples and suggest improvements"""
  try:
    print(f"Starting ROI coverage analysis for template: {template_id}")
    tpath = os.path.join(TEMPLATES_DIR, f"{template_id}.json")
    if not os.path.exists(tpath):
      return JSONResponse({"error": "unknown template"}, status_code=400)

    cfg = json.load(open(tpath, "r", encoding="utf-8"))
    canon_path = os.path.join(TEMPLATES_DIR, os.path.basename(cfg["canonical_image"]))
    
    canonical = cv2.imread(canon_path, cv2.IMREAD_COLOR)
    if canonical is None:
      return JSONResponse({"error": "canonical not loadable"}, status_code=500)

    analysis_results = []
    
    for i, file in enumerate(files):
      print(f"Analyzing file {i+1}/{len(files)}: {file.filename}")
      
      try:
        # Load and align image
        data = await file.read()
        file_bytes = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
          pil = Image.open(io.BytesIO(data)).convert("RGB")
          img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        
        print(f"Image {i+1} loaded: {img.shape}")
        
        # Try alignment, but use a more permissive approach for analysis
        aligned = align_to_canonical(img, canonical)
        
        # If alignment failed (returned original), try to scale the image to canonical size
        if aligned.shape == img.shape:
          print(f"Alignment failed, scaling image to canonical size")
          canonical_h, canonical_w = canonical.shape[:2]
          aligned = cv2.resize(img, (canonical_w, canonical_h))
        
        print(f"Image {i+1} processed: {aligned.shape}")
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        
        # Use canonical dimensions for coordinate conversion
        # The ROI coordinates are defined relative to the canonical template
        H, W = canonical.shape[:2]  # Use canonical dimensions
        print(f"Using canonical dimensions for ROI conversion: {W}x{H}")
        
      except Exception as e:
        print(f"Error processing image {i+1}: {str(e)}")
        continue
      
      # Convert ROIs to pixels
      def rel_to_px_analysis(r):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        print(f"Converting ROI {r['name']}: original coords ({x}, {y}, {w}, {h})")
        print(f"Conversion check: x in 0-1? {0 <= x <= 1}, y in 0-1? {0 <= y <= 1}, w in 0-1? {0 < w <= 1}, h in 0-1? {0 < h <= 1}")
        
        if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
          print(f"Converting relative to pixel using W={W}, H={H}")
          x, y, w, h = round(x*W), round(y*H), round(w*W), round(h*H)
          print(f"Converted to pixels: ({x}, {y}, {w}, {h})")
        else:
          print(f"Coordinates appear to already be in pixels")
        
        # Ensure all coordinates are integers
        x, y, w, h = int(x), int(y), int(w), int(h)
        print(f"Final coordinates: ({x}, {y}, {w}, {h})")
        
        # Create new dict with converted coordinates - don't let **r override them!
        result = r.copy()
        result.update({"x": x, "y": y, "w": w, "h": h})
        return result
      
      rois_px = [rel_to_px_analysis(r) for r in cfg["rois"]]
      print(f"Converted {len(rois_px)} ROIs")
      
      # Analyze text distribution for each ROI
      file_analysis = {"filename": file.filename, "rois": []}
      
      for roi_idx, roi in enumerate(rois_px):
        try:
          print(f"Processing ROI {roi_idx+1}: {roi['name']}")
          # Access the converted coordinates correctly
          x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]  # These should be the converted pixel values
          print(f"ROI coordinates: x={x}, y={y}, w={w}, h={h}")
          
          # Debug: print the full roi object to see what's in it
          print(f"Full ROI object: {roi}")
          
          # Get larger area around ROI to analyze text distribution
          expanded_x = max(0, x - w//2)
          expanded_y = max(0, y - h//2)
          expanded_w = min(W - expanded_x, w * 2)
          expanded_h = min(H - expanded_y, h * 2)
          
          # Ensure all expanded coordinates are integers
          expanded_x, expanded_y = int(expanded_x), int(expanded_y)
          expanded_w, expanded_h = int(expanded_w), int(expanded_h)
          
          print(f"Expanded coordinates: x={expanded_x}, y={expanded_y}, w={expanded_w}, h={expanded_h}")
          
          if expanded_w <= 0 or expanded_h <= 0:
            print(f"Invalid expanded dimensions for ROI {roi['name']}")
            continue
          
          expanded_patch = gray[expanded_y:expanded_y+expanded_h, expanded_x:expanded_x+expanded_w]
          
        except Exception as roi_error:
          print(f"Error processing ROI {roi.get('name', 'unknown')}: {str(roi_error)}")
          continue
        
        # Find text regions using adaptive threshold
        binary = cv2.adaptiveThreshold(expanded_patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours (potential text)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        text_contours = []
        for contour in contours:
          area = cv2.contourArea(contour)
          if 50 < area < expanded_w * expanded_h * 0.3:  # Reasonable text size
            text_contours.append(contour)
        
        # Calculate text bounds within expanded area
        if text_contours:
          all_points = np.vstack(text_contours)
          text_x, text_y, text_w, text_h = cv2.boundingRect(all_points)
          
          # Convert back to original image coordinates
          absolute_text_x = expanded_x + text_x
          absolute_text_y = expanded_y + text_y
          
          # Calculate how well current ROI covers the text
          roi_coverage = calculate_overlap_ratio(
            (x, y, w, h), 
            (absolute_text_x, absolute_text_y, text_w, text_h)
          )
          
          file_analysis["rois"].append({
            "name": roi["name"],
            "current_roi": {"x": x, "y": y, "w": w, "h": h},
            "detected_text_bounds": {"x": absolute_text_x, "y": absolute_text_y, "w": text_w, "h": text_h},
            "coverage_ratio": round(roi_coverage, 3)
          })
        else:
          file_analysis["rois"].append({
            "name": roi["name"],
            "current_roi": {"x": x, "y": y, "w": w, "h": h},
            "detected_text_bounds": None,
            "coverage_ratio": 0.0
          })
      
      analysis_results.append(file_analysis)
    
    # Calculate suggestions based on all samples
    suggestions = calculate_roi_suggestions(analysis_results, cfg)
    
    return {
      "template_id": template_id,
      "samples_analyzed": len(files),
      "analysis_results": analysis_results,
      "suggestions": suggestions
    }
    
  except Exception as e:
    print(f"Error in ROI analysis: {str(e)}")
    import traceback
    traceback.print_exc()
    return JSONResponse({"error": str(e)}, status_code=500)

def calculate_overlap_ratio(roi1, roi2):
  """Calculate how much roi1 overlaps with roi2 (0.0 to 1.0)"""
  x1, y1, w1, h1 = roi1
  x2, y2, w2, h2 = roi2
  
  # Calculate intersection
  ix1 = max(x1, x2)
  iy1 = max(y1, y2)
  ix2 = min(x1 + w1, x2 + w2)
  iy2 = min(y1 + h1, y2 + h2)
  
  if ix1 >= ix2 or iy1 >= iy2:
    return 0.0
  
  intersection_area = (ix2 - ix1) * (iy2 - iy1)
  roi2_area = w2 * h2
  
  return intersection_area / roi2_area if roi2_area > 0 else 0.0

def calculate_roi_suggestions(analysis_results, template_config):
  """Calculate suggested ROI adjustments based on multiple sample analysis"""
  suggestions = []
  
  # Group results by ROI name
  roi_groups = {}
  for result in analysis_results:
    for roi_analysis in result["rois"]:
      roi_name = roi_analysis["name"]
      if roi_name not in roi_groups:
        roi_groups[roi_name] = []
      roi_groups[roi_name].append(roi_analysis)
  
  for roi_name, roi_analyses in roi_groups.items():
    # Find the original ROI config
    original_roi = next((r for r in template_config["rois"] if r["name"] == roi_name), None)
    if not original_roi:
      continue
    
    # Calculate statistics
    coverage_ratios = [r["coverage_ratio"] for r in roi_analyses if r["detected_text_bounds"]]
    avg_coverage = sum(coverage_ratios) / len(coverage_ratios) if coverage_ratios else 0
    
    # Find common text bounds across samples
    text_bounds = [r["detected_text_bounds"] for r in roi_analyses if r["detected_text_bounds"]]
    
    if text_bounds:
      # Calculate union of all text bounds
      min_x = min(b["x"] for b in text_bounds)
      min_y = min(b["y"] for b in text_bounds)
      max_x = max(b["x"] + b["w"] for b in text_bounds)
      max_y = max(b["y"] + b["h"] for b in text_bounds)
      
      # Convert back to relative coordinates
      canvas_w = template_config["canvas_width"]
      canvas_h = template_config["canvas_height"]
      
      suggested_roi = {
        "x": round(min_x / canvas_w, 4),
        "y": round(min_y / canvas_h, 4),
        "w": round((max_x - min_x) / canvas_w, 4),
        "h": round((max_y - min_y) / canvas_h, 4)
      }
      
      suggestions.append({
        "roi_name": roi_name,
        "current_coverage": round(avg_coverage, 3),
        "samples_with_text": len(text_bounds),
        "original_roi": original_roi,
        "suggested_roi": suggested_roi,
        "improvement_needed": avg_coverage < 0.8
      })
  
  return suggestions

@app.post("/submit-correction")
async def submit_correction(
    field_name: str = Form(...),
    predicted_text: str = Form(...),
    correct_text: str = Form(...),
    confidence: float = Form(0.0),
    user_id: str = Form("api_user"),
    file: UploadFile = File(...)
):
    """Submit a correction for model training"""
    try:
        # Save the image temporarily
        temp_path = f"temp_corrections/{file.filename}"
        os.makedirs("temp_corrections", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Log the correction
        learning_pipeline.log_correction(
            image_path=temp_path,
            field_type=field_name,
            predicted_text=predicted_text,
            correct_text=correct_text,
            confidence_score=confidence,
            user_id=user_id
        )
        
        return JSONResponse({
            "status": "success",
            "message": "Correction logged successfully",
            "field_name": field_name,
            "correction": f"'{predicted_text}' -> '{correct_text}'"
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/training-status")
async def get_training_status():
    """Get current training status and insights"""
    try:
        insights = learning_pipeline.get_performance_insights()
        
        # Check how many samples we have for retraining
        import sqlite3
        conn = sqlite3.connect(learning_pipeline.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM corrections WHERE used_for_training = FALSE")
        unused_samples = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM corrections")
        total_samples = cursor.fetchone()[0]
        
        conn.close()
        
        return JSONResponse({
            "status": "success",
            "training_ready": unused_samples >= learning_pipeline.min_samples_for_retrain,
            "unused_samples": unused_samples,
            "total_samples": total_samples,
            "min_samples_needed": learning_pipeline.min_samples_for_retrain,
            "insights": insights
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/trigger-training")
async def trigger_training():
    """Trigger model retraining with collected corrections"""
    try:
        # Check if we have enough samples
        import sqlite3
        conn = sqlite3.connect(learning_pipeline.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM corrections WHERE used_for_training = FALSE")
        unused_samples = cursor.fetchone()[0]
        conn.close()
        
        if unused_samples < learning_pipeline.min_samples_for_retrain:
            return JSONResponse({
                "status": "error",
                "message": f"Not enough samples. Need {learning_pipeline.min_samples_for_retrain}, have {unused_samples}"
            })
        
        # Prepare training data
        training_metadata = learning_pipeline.prepare_training_data()
        
        # Mark samples as used
        learning_pipeline.mark_samples_as_used()
        
        return JSONResponse({
            "status": "success",
            "message": "Training data prepared. Run the training script to fine-tune the model.",
            "training_samples": training_metadata["sample_count"],
            "next_steps": [
                "1. Install training dependencies: pip install torch transformers evaluate datasets",
                "2. Run training: python train_trocr.py",
                "3. Update model path in htr.py to use fine-tuned model"
            ]
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )