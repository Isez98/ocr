# Snap-to-Ink Auto-Refine

Automatically tighten OCR template ROI boxes by detecting ink bounds and applying type-specific padding. This feature reduces template creation time and improves OCR accuracy by ensuring ROIs are optimally sized around actual text content.

## Features

### Core Functionality
- **Adaptive Thresholding**: Converts ROI to binary image to detect ink
- **Morphological Cleaning**: Removes noise with 3×3 morphological opening
- **Projection Analysis**: Uses row/column sums to find tight ink bounds
- **Type-Specific Padding**: Applies different padding rules based on field type

### Supported Field Types
- **text** (names/addresses): +10–20px vertical for ascenders/accents
- **digits**: +4–8px vertical, +8–16px horizontal (people space digits)
- **date**: +4–8px vertical, +8–16px horizontal
- **phone**: +4–8px vertical, +8–16px horizontal  
- **currency**: Extra right padding for decimals/symbols

### Base Padding
- Automatically calculated as ~1% of canvas size
- Minimum 10px, maximum 20px
- Ensures tall letters and diacritics aren't clipped

## Usage

### 1. Interactive ROI Selector (Recommended)
```bash
python roi_selector.py path/to/canonical_image.png
```

When creating each ROI:
- Enter ROI name and type
- Choose **y** when prompted "Apply snap-to-ink auto-refine?"
- Review refined coordinates and confirm

Batch refinement:
- Press **'a'** to apply snap-to-ink to all existing ROIs
- Review changes and confirm application

### 2. Command Line Tool
```bash
# Interactive mode with preview
python refine_template.py templates/form_v1.json

# Auto-apply without preview
python refine_template.py templates/form_v1.json --auto-apply

# Specify custom image path
python refine_template.py templates/form_v1.json --image-path path/to/canonical.png

# Save to different file
python refine_template.py templates/form_v1.json --output templates/form_v1_refined.json
```

### 3. REST API
```bash
# Apply snap-to-ink via API
curl -X POST "http://localhost:8000/snap-to-ink" \
     -F "template_id=form_v1"
```

Response includes:
- Summary of changes
- Backup file location
- Base64-encoded preview image
- Change statistics

### 4. Web Interface
Open `snap_to_ink_ui.html` in your browser and connect to the OCR service:
- Enter template ID
- Click "Apply Snap-to-Ink Refinement"
- View preview and change summary

## API Integration

### FastAPI Endpoint
```python
@app.post("/snap-to-ink")
async def snap_to_ink_endpoint(template_id: str = Form(...)):
    """Apply snap-to-ink auto-refinement to an existing template"""
```

### Python Module
```python
from snap_to_ink import snap_template_to_ink, snap_roi_to_ink

# Refine entire template
refined_template, preview_image = snap_template_to_ink(
    image_path, template, preview_mode=True
)

# Refine single ROI
refined_roi = snap_roi_to_ink(
    gray_image, roi, canvas_width, canvas_height
)
```

## Algorithm Details

### 1. Preprocessing
```python
# Adaptive threshold to create binary image
binary = cv2.adaptiveThreshold(
    roi_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV, 11, 2
)

# Morphological opening to clean noise
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
```

### 2. Ink Detection
```python
# Row and column projection sums
row_sums = np.sum(binary_image, axis=1)
col_sums = np.sum(binary_image, axis=0)

# Find first/last non-zero positions
y_min, y_max = first_last_nonzero(row_sums)
x_min, x_max = first_last_nonzero(col_sums)
```

### 3. Padding Application
```python
# Type-specific padding rules
if roi_type in ["digits", "date", "phone"]:
    padding = {
        "vertical": base_pad + 6,
        "horizontal": base_pad + 12
    }
elif roi_type == "currency":
    padding = {
        "vertical": base_pad + 6,
        "horizontal_left": base_pad + 8,
        "horizontal_right": base_pad + 20  # Extra for decimals
    }
else:  # text
    padding = {
        "vertical": base_pad + 15,  # For ascenders/accents
        "horizontal": base_pad + 10
    }
```

## Benefits

1. **Improved Accuracy**: Tighter ROIs reduce noise and improve OCR confidence
2. **Reduced Template Size**: Smaller ROIs process faster and use less memory
3. **Consistent Quality**: Automated refinement ensures consistent ROI sizing
4. **Time Savings**: Reduces manual template creation and adjustment time
5. **Type Awareness**: Field-specific padding optimizes for different content types

## Files

- `snap_to_ink.py` - Core implementation module
- `refine_template.py` - Command line tool
- `snap_to_ink_ui.html` - Web interface
- `test_snap_to_ink.py` - Test suite with synthetic data
- `roi_selector.py` - Enhanced ROI selector with snap-to-ink integration

## Testing

Run the test suite:
```bash
python test_snap_to_ink.py
```

The test creates synthetic data and demonstrates:
- Individual ROI refinement
- Full template processing
- Type-specific padding rules
- Area reduction calculations
- Preview image generation

## Safety Features

- **Automatic Backups**: Original templates are backed up before modification
- **Minimum Size Limits**: Prevents ROIs from becoming too small
- **Canvas Bounds**: Ensures refined ROIs stay within image boundaries
- **Fallback Handling**: Returns original ROI if ink detection fails
- **Preview Mode**: Visual confirmation before applying changes