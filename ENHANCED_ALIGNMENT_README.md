# Enhanced OCR Alignment System

This document describes the enhanced alignment system for the OCR module, which provides perfect warp correction using fiducial markers and stable anchor detection.

## Overview

The enhanced alignment system improves OCR accuracy by ensuring perfect alignment between scanned documents and their canonical templates. It provides two main alignment methods:

1. **Fiducial Marker Alignment** (Preferred) - Uses ArUco markers for precise alignment
2. **Stable Anchor Alignment** (Fallback) - Uses logos, text keywords, and geometric shapes

## Features

- **ArUco Fiducial Markers**: Precise alignment using DICT_4X4_50 markers
- **Stable Anchor Detection**: Automatic detection of logos, text anchors, and checkbox corners
- **Robust Homography Computation**: RANSAC-based outlier rejection
- **Quality Validation**: Automatic assessment of alignment quality
- **Fallback Strategy**: Graceful degradation when primary methods fail

## Installation

1. Install the enhanced OpenCV package:
```bash
pip install opencv-contrib-python==4.10.0.84
```

2. Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Quick Start

### Method 1: Using Fiducial Markers (Recommended)

1. **Add fiducial markers to your template document** (if you control the print):
   - Place four tiny ArUco markers (DICT_4X4_50) at the corners of your form
   - Markers should be small (e.g., 5-10mm) and positioned in margins
   - Use marker IDs 0, 1, 2, 3 for the four corners

2. **Create a test template with fiducials**:
```bash
python template_manager.py create-test test_fiducials.png --width 1653 --height 2339
```

3. **Scan your canonical document** and save the fiducial positions:
```bash
python template_manager.py add-fiducials templates/form_v1.json canonical_form_with_fiducials.png
```

4. **Process new scans** with enhanced alignment:
```bash
python snap_to_ink.py scanned_form.png templates/form_v1.json
```

### Method 2: Using Stable Anchors (No Print Control)

1. **Add stable anchors to your template**:
```bash
python template_manager.py add-anchors templates/form_v1.json canonical_form.png
```

2. **Process scans** with anchor-based alignment:
```bash
python snap_to_ink.py scanned_form.png templates/form_v1.json --no-fiducials
```

## Template Management

### Adding Fiducial Markers

To add fiducial marker support to an existing template:

```bash
python template_manager.py add-fiducials templates/form_v1.json canonical_image.png
```

This will:
- Detect ArUco markers in the canonical image
- Record their pixel centers in the template
- Create a backup of the original template

### Adding Stable Anchors

To add stable anchor points for fallback alignment:

```bash
python template_manager.py add-anchors templates/form_v1.json canonical_image.png --names logo header_text checkbox1 checkbox2
```

### Visualizing Alignment Points

To see all alignment points in your template:

```bash
python template_manager.py visualize templates/form_v1.json canonical_image.png --output alignment_viz.png
```

### Testing Alignment Robustness

To test alignment across multiple images:

```bash
python template_manager.py test templates/form_v1.json scan1.png scan2.png scan3.png --output-dir test_results
```

## API Usage

### Basic Alignment

```python
from alignment_utils import AlignmentEngine

# Initialize engine
engine = AlignmentEngine()

# Load images and template
runtime_image = cv2.imread('scanned_form.png')
with open('templates/form_v1.json', 'r') as f:
    template = json.load(f)

# Extract canonical points
canonical_points = {}
if 'fiducials' in template:
    canonical_points['fiducials'] = template['fiducials']
if 'anchors' in template:
    canonical_points['anchors'] = template['anchors']

# Compute homography
homography = engine.compute_alignment_homography(
    runtime_image, canonical_points, 
    use_fiducials=True
)

if homography is not None:
    # Apply alignment
    target_size = (template['canvas_width'], template['canvas_height'])
    aligned_image = engine.apply_alignment(runtime_image, homography, target_size)
```

### Enhanced Snap-to-Ink Processing

```python
from snap_to_ink import snap_template_to_ink, validate_alignment_quality

# Process with alignment
refined_template, preview_image = snap_template_to_ink(
    'scanned_form.png',
    template,
    preview_mode=True,
    enable_alignment=True,
    use_fiducials=True
)

# Check alignment quality
original_image = cv2.imread('scanned_form.png')
aligned_image = cv2.imread('aligned_form.png')
quality = validate_alignment_quality(original_image, aligned_image, template)
print(f"Alignment quality: {quality['overall_quality']:.2f}")
```

## Template Structure

Enhanced templates include additional fields for alignment:

```json
{
  "id": "form_v1",
  "canvas_width": 1653,
  "canvas_height": 2339,
  "canonical_image": "templates/canonical_form_v1.png",
  "fiducials": {
    "0": [150.5, 150.5],
    "1": [1503.5, 150.5],
    "2": [150.5, 2189.5],
    "3": [1503.5, 2189.5]
  },
  "anchors": {
    "logo": [826.5, 100.0],
    "header_text": [826.5, 200.0],
    "checkbox1": [100.0, 500.0],
    "checkbox2": [100.0, 600.0]
  },
  "rois": [
    // ... existing ROI definitions
  ]
}
```

## Alignment Strategy

The system follows this alignment strategy:

1. **Try Fiducial Markers First** (if `use_fiducials=True` and fiducials present):
   - Detect ArUco markers in runtime image
   - Match to canonical marker positions
   - Compute homography if ≥4 markers found

2. **Fall Back to Stable Anchors**:
   - Detect logos using template matching
   - Find text anchors using morphological operations
   - Detect checkbox corners using Canny + Hough
   - Require ≥3 distinct anchors for alignment

3. **Validate Alignment Quality**:
   - Check ink detection improvement in ROIs
   - Assess edge alignment patterns
   - Use aligned image only if quality > 0.5

4. **Fail Safely**:
   - If alignment fails, process original image
   - Log detailed error information
   - Provide quality feedback

## Best Practices

### For Fiducial Markers:
- Use small markers (5-10mm) in document corners
- Ensure markers don't interfere with content
- Use high contrast (black markers on white background)
- Test marker detection under various lighting conditions

### For Stable Anchors:
- Choose distinctive, high-contrast features
- Prefer printed logos over handwritten content
- Use geometric elements (checkboxes, lines) when available
- Avoid areas that may be covered by writing

### General:
- Always test alignment with multiple sample images
- Monitor alignment quality metrics
- Keep backup templates without alignment data
- Document canonical image requirements

## Troubleshooting

### "Insufficient fiducial markers detected"
- Check marker print quality and contrast
- Ensure markers are not occluded
- Try different lighting conditions
- Verify ArUco dictionary (DICT_4X4_50)

### "Insufficient anchors detected"
- Increase image contrast/resolution
- Try different anchor detection parameters
- Add more distinctive features to template
- Check for image distortion/skew

### "Alignment quality insufficient"
- Verify canonical image matches template exactly
- Check for significant document variations
- Consider updating canonical points
- Try manual alignment verification

### Performance Issues:
- Resize large images before processing
- Cache template data for repeated use
- Optimize anchor detection parameters
- Consider ROI-based processing for large documents

## Command Line Options

### snap_to_ink.py
```bash
python snap_to_ink.py <image> <template> [options]

Options:
  --no-alignment     Disable image alignment
  --no-fiducials     Disable fiducial marker detection
```

### template_manager.py
```bash
python template_manager.py <command> [args]

Commands:
  add-fiducials      Add fiducial markers to template
  add-anchors        Add stable anchors to template  
  create             Create new template with fiducials
  visualize          Visualize template alignment points
  test               Test alignment robustness
  create-test        Create test image with fiducials
```

## Integration with Existing Workflow

The enhanced alignment system is designed to be backward compatible:

1. **Existing templates** continue to work without modification
2. **Legacy processing** available with `--no-alignment` flag
3. **Gradual migration** - add alignment features incrementally
4. **Quality validation** - automatic fallback to original processing

## Future Enhancements

Planned improvements include:

- **Machine Learning-Based Matching**: Neural network feature matching
- **Multi-Scale Detection**: Improved robustness to scale variations
- **Rotation Correction**: Automatic skew detection and correction
- **Template Learning**: Automatic anchor point discovery
- **Performance Optimization**: GPU acceleration for large documents