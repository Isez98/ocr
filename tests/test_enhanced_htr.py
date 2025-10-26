#!/usr/bin/env python3
"""
Test script for the enhanced HTR pipeline
Tests all sample forms and provides detailed analysis
"""

import json
import os
import cv2
import numpy as np
from PIL import Image
from htr import align_to_canonical, deskew_and_binarize, crop, htr_read, postprocess

def test_form_processing(sample_path: str, template_id: str = "form_v1"):
    """Process a single form and return results"""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(sample_path)}")
    print(f"{'='*60}")
    
    # Load template configuration
    template_path = f"templates/{template_id}.json"
    with open(template_path, 'r') as f:
        cfg = json.load(f)
    
    # Load canonical image
    canon_path = os.path.join("templates", os.path.basename(cfg["canonical_image"]))
    canonical = cv2.imread(canon_path, cv2.IMREAD_COLOR)
    
    # Load and process the sample image
    img = cv2.imread(sample_path, cv2.IMREAD_COLOR)
    if img is None:
        pil = Image.open(sample_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    
    print(f"Original image size: {img.shape[:2]}")
    print(f"Canonical size: {canonical.shape[:2]}")
    
    # Align to canonical
    aligned = align_to_canonical(img, canonical)
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    print(f"Aligned image size: {gray.shape}")
    
    results = []
    total_confidence = 0
    processed_fields = 0
    
    for i, roi in enumerate(cfg["rois"]):
        field_name = roi["name"]
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        field_type = roi.get("type", "text")
        
        # Extract patch
        patch = crop(gray, x, y, w, h)
        
        print(f"\nField: {field_name} ({field_type})")
        print(f"  ROI: ({x}, {y}, {w}, {h})")
        print(f"  Patch size: {patch.shape}")
        
        if patch.shape[0] < 10 or patch.shape[1] < 10:
            print(f"  ⚠️ Patch too small, skipping")
            results.append({
                "name": field_name,
                "text": "",
                "confidence": 0.0,
                "status": "skipped_small"
            })
            continue
        
        # Binarize and prepare for HTR
        binimg = deskew_and_binarize(patch)
        rgb_img = cv2.cvtColor(binimg, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        # Run HTR
        try:
            raw_text, confidence = htr_read(pil_img, field_type)
            processed_text = postprocess(raw_text, field_type)
            
            print(f"  Raw HTR: '{raw_text}'")
            print(f"  Processed: '{processed_text}'")
            print(f"  Confidence: {confidence:.3f}")
            
            if confidence > 0.5:
                print(f"  ✅ Good confidence")
            elif confidence > 0.3:
                print(f"  ⚠️ Medium confidence")
            else:
                print(f"  ❌ Low confidence")
            
            results.append({
                "name": field_name,
                "type": field_type,
                "raw_text": raw_text,
                "text": processed_text,
                "confidence": confidence,
                "status": "processed"
            })
            
            total_confidence += confidence
            processed_fields += 1
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "name": field_name,
                "text": "",
                "confidence": 0.0,
                "status": f"error: {str(e)}"
            })
    
    # Summary
    avg_confidence = total_confidence / processed_fields if processed_fields > 0 else 0
    print(f"\n{'='*60}")
    print(f"SUMMARY for {os.path.basename(sample_path)}")
    print(f"{'='*60}")
    print(f"Fields processed: {processed_fields}/{len(cfg['rois'])}")
    print(f"Average confidence: {avg_confidence:.3f}")
    
    high_conf = sum(1 for r in results if r.get('confidence', 0) > 0.5)
    med_conf = sum(1 for r in results if 0.3 <= r.get('confidence', 0) <= 0.5)
    low_conf = sum(1 for r in results if 0 < r.get('confidence', 0) < 0.3)
    
    print(f"High confidence (>0.5): {high_conf}")
    print(f"Medium confidence (0.3-0.5): {med_conf}")
    print(f"Low confidence (<0.3): {low_conf}")
    
    return {
        "sample": os.path.basename(sample_path),
        "fields": results,
        "summary": {
            "processed_fields": processed_fields,
            "total_fields": len(cfg['rois']),
            "avg_confidence": avg_confidence,
            "high_confidence_count": high_conf,
            "medium_confidence_count": med_conf,
            "low_confidence_count": low_conf
        }
    }

def main():
    print("Enhanced HTR Pipeline Test")
    print("=" * 60)
    
    # Test with all sample forms
    samples_dir = "../samples"
    sample_files = [f for f in os.listdir(samples_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    sample_files.sort()
    
    all_results = []
    
    for sample_file in sample_files:
        sample_path = os.path.join(samples_dir, sample_file)
        result = test_form_processing(sample_path)
        all_results.append(result)
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    total_samples = len(all_results)
    total_fields = sum(r['summary']['total_fields'] for r in all_results)
    total_processed = sum(r['summary']['processed_fields'] for r in all_results)
    overall_avg_conf = sum(r['summary']['avg_confidence'] for r in all_results) / total_samples if total_samples > 0 else 0
    
    print(f"Samples processed: {total_samples}")
    print(f"Total fields: {total_fields}")
    print(f"Successfully processed: {total_processed}")
    print(f"Overall average confidence: {overall_avg_conf:.3f}")
    
    # Best and worst performing fields
    all_field_results = []
    for result in all_results:
        for field in result['fields']:
            if field.get('confidence', 0) > 0:
                all_field_results.append(field)
    
    if all_field_results:
        # Sort by confidence
        all_field_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        print(f"\nBest performing fields:")
        for field in all_field_results[:5]:
            print(f"  {field['name']} ({field['type']}): '{field['text']}' - {field['confidence']:.3f}")
        
        print(f"\nWorst performing fields:")
        for field in all_field_results[-5:]:
            print(f"  {field['name']} ({field['type']}): '{field['text']}' - {field['confidence']:.3f}")
    
    # Field type performance
    print(f"\nPerformance by field type:")
    field_types = {}
    for result in all_results:
        for field in result['fields']:
            ftype = field.get('type', 'unknown')
            if ftype not in field_types:
                field_types[ftype] = []
            if field.get('confidence', 0) > 0:
                field_types[ftype].append(field['confidence'])
    
    for ftype, confidences in field_types.items():
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            print(f"  {ftype}: {avg_conf:.3f} (from {len(confidences)} fields)")

if __name__ == "__main__":
    main()