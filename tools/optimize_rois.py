#!/usr/bin/env python3
"""
ROI Optimization Tool
Analyzes current performance and suggests ROI adjustments
"""

import json
import os
import cv2
import numpy as np
from PIL import Image
from htr import align_to_canonical, deskew_and_binarize, crop, htr_read

def analyze_roi_performance(sample_path: str, template_id: str = "form_v1"):
    """Analyze each ROI and suggest optimizations"""
    
    # Load template and canonical
    template_path = f"templates/{template_id}.json"
    with open(template_path, 'r') as f:
        cfg = json.load(f)
    
    canon_path = os.path.join("templates", os.path.basename(cfg["canonical_image"]))
    canonical = cv2.imread(canon_path, cv2.IMREAD_COLOR)
    
    # Load and align sample
    img = cv2.imread(sample_path, cv2.IMREAD_COLOR)
    if img is None:
        pil = Image.open(sample_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    
    aligned = align_to_canonical(img, canonical)
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    
    suggestions = []
    
    for roi in cfg["rois"]:
        field_name = roi["name"]
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        field_type = roi.get("type", "text")
        
        # Current ROI
        current_patch = crop(gray, x, y, w, h)
        
        # Test current ROI
        if current_patch.shape[0] >= 10 and current_patch.shape[1] >= 10:
            binimg = deskew_and_binarize(current_patch)
            rgb_img = cv2.cvtColor(binimg, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(rgb_img)
            current_text, current_conf = htr_read(pil_img, field_type)
        else:
            current_text, current_conf = "", 0.0
        
        suggestion = {
            "field": field_name,
            "type": field_type,
            "current_roi": {"x": x, "y": y, "w": w, "h": h},
            "current_confidence": current_conf,
            "current_text": current_text,
            "optimizations": []
        }
        
        # Test variations
        variations = []
        
        # 1. Expand horizontally
        expanded_w = min(w + 20, gray.shape[1] - x)
        if expanded_w > w:
            exp_patch = crop(gray, x, y, expanded_w, h)
            if exp_patch.shape[0] >= 10 and exp_patch.shape[1] >= 10:
                binimg = deskew_and_binarize(exp_patch)
                rgb_img = cv2.cvtColor(binimg, cv2.COLOR_GRAY2RGB)
                pil_img = Image.fromarray(rgb_img)
                exp_text, exp_conf = htr_read(pil_img, field_type)
                variations.append(("expand_width", {"x": x, "y": y, "w": expanded_w, "h": h}, exp_conf, exp_text))
        
        # 2. Expand vertically
        expanded_h = min(h + 10, gray.shape[0] - y)
        if expanded_h > h:
            exp_patch = crop(gray, x, y, w, expanded_h)
            if exp_patch.shape[0] >= 10 and exp_patch.shape[1] >= 10:
                binimg = deskew_and_binarize(exp_patch)
                rgb_img = cv2.cvtColor(binimg, cv2.COLOR_GRAY2RGB)
                pil_img = Image.fromarray(rgb_img)
                exp_text, exp_conf = htr_read(pil_img, field_type)
                variations.append(("expand_height", {"x": x, "y": y, "w": w, "h": expanded_h}, exp_conf, exp_text))
        
        # 3. Shift left (if text might start earlier)
        if x >= 10:
            shifted_x = x - 10
            shifted_w = w + 10
            shift_patch = crop(gray, shifted_x, y, shifted_w, h)
            if shift_patch.shape[0] >= 10 and shift_patch.shape[1] >= 10:
                binimg = deskew_and_binarize(shift_patch)
                rgb_img = cv2.cvtColor(binimg, cv2.COLOR_GRAY2RGB)
                pil_img = Image.fromarray(rgb_img)
                shift_text, shift_conf = htr_read(pil_img, field_type)
                variations.append(("shift_left", {"x": shifted_x, "y": y, "w": shifted_w, "h": h}, shift_conf, shift_text))
        
        # 4. Shift up (if text might be higher)
        if y >= 5:
            shifted_y = y - 5
            shifted_h = h + 5
            shift_patch = crop(gray, x, shifted_y, w, shifted_h)
            if shift_patch.shape[0] >= 10 and shift_patch.shape[1] >= 10:
                binimg = deskew_and_binarize(shift_patch)
                rgb_img = cv2.cvtColor(binimg, cv2.COLOR_GRAY2RGB)
                pil_img = Image.fromarray(rgb_img)
                shift_text, shift_conf = htr_read(pil_img, field_type)
                variations.append(("shift_up", {"x": x, "y": shifted_y, "w": w, "h": shifted_h}, shift_conf, shift_text))
        
        # Find best variation
        best_variation = None
        best_improvement = 0
        
        for var_name, var_roi, var_conf, var_text in variations:
            improvement = var_conf - current_conf
            if improvement > best_improvement:
                best_improvement = improvement
                best_variation = (var_name, var_roi, var_conf, var_text, improvement)
        
        if best_variation:
            var_name, var_roi, var_conf, var_text, improvement = best_variation
            suggestion["optimizations"].append({
                "type": var_name,
                "new_roi": var_roi,
                "new_confidence": var_conf,
                "new_text": var_text,
                "improvement": improvement,
                "recommended": improvement > 0.1  # Recommend if significant improvement
            })
        
        suggestions.append(suggestion)
    
    return suggestions

def generate_optimized_template(suggestions, template_id="form_v1"):
    """Generate an optimized template based on suggestions"""
    
    # Load original template
    template_path = f"templates/{template_id}.json"
    with open(template_path, 'r') as f:
        cfg = json.load(f)
    
    optimized_cfg = cfg.copy()
    optimized_cfg["id"] = f"{template_id}_optimized"
    
    changes_made = 0
    
    for i, roi in enumerate(optimized_cfg["rois"]):
        field_name = roi["name"]
        
        # Find suggestion for this field
        suggestion = next((s for s in suggestions if s["field"] == field_name), None)
        if suggestion and suggestion["optimizations"]:
            # Use the best recommended optimization
            best_opt = next((opt for opt in suggestion["optimizations"] if opt["recommended"]), None)
            if best_opt:
                old_roi = roi.copy()
                roi.update(best_opt["new_roi"])
                changes_made += 1
                print(f"Optimized {field_name}: {old_roi} -> {best_opt['new_roi']} (+{best_opt['improvement']:.3f} confidence)")
    
    if changes_made > 0:
        output_path = f"templates/{template_id}_optimized.json"
        with open(output_path, 'w') as f:
            json.dump(optimized_cfg, f, indent=2)
        print(f"\nOptimized template saved to {output_path}")
        print(f"Made {changes_made} improvements")
    else:
        print("No significant improvements found - current ROIs are already well optimized!")
    
    return optimized_cfg

def main():
    print("ROI Optimization Analysis")
    print("=" * 50)
    
    # Analyze all samples
    samples_dir = "../samples"
    sample_files = [f for f in os.listdir(samples_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    all_suggestions = []
    
    for sample_file in sample_files:
        print(f"\nAnalyzing {sample_file}...")
        sample_path = os.path.join(samples_dir, sample_file)
        suggestions = analyze_roi_performance(sample_path)
        all_suggestions.extend(suggestions)
    
    # Aggregate suggestions across all samples
    field_performance = {}
    for suggestion in all_suggestions:
        field_name = suggestion["field"]
        if field_name not in field_performance:
            field_performance[field_name] = {
                "confidences": [],
                "optimizations": []
            }
        
        field_performance[field_name]["confidences"].append(suggestion["current_confidence"])
        field_performance[field_name]["optimizations"].extend(suggestion["optimizations"])
    
    print(f"\n{'=' * 50}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'=' * 50}")
    
    for field_name, data in field_performance.items():
        avg_conf = sum(data["confidences"]) / len(data["confidences"])
        recommended_opts = [opt for opt in data["optimizations"] if opt.get("recommended", False)]
        
        print(f"\n{field_name}:")
        print(f"  Average confidence: {avg_conf:.3f}")
        print(f"  Recommended optimizations: {len(recommended_opts)}")
        
        if recommended_opts:
            avg_improvement = sum(opt["improvement"] for opt in recommended_opts) / len(recommended_opts)
            print(f"  Average improvement potential: +{avg_improvement:.3f}")
            
            # Show most common optimization type
            opt_types = [opt["type"] for opt in recommended_opts]
            most_common = max(set(opt_types), key=opt_types.count) if opt_types else None
            if most_common:
                print(f"  Most common optimization: {most_common}")
    
    # Generate optimized template using first sample as reference
    if sample_files:
        first_sample = os.path.join(samples_dir, sample_files[0])
        print(f"\nGenerating optimized template based on {sample_files[0]}...")
        reference_suggestions = analyze_roi_performance(first_sample)
        generate_optimized_template(reference_suggestions)

if __name__ == "__main__":
    main()