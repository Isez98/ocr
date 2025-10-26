# OCR Project Navigation Guide

## 🚀 Quick Start Commands

### Training a New Model
```bash
# CPU training (recommended for AMD RX 6800)
cd training/
python train_gpu.py  # Will fallback to CPU if GPU issues

# Check training progress
cd ../models/
python test_model.py [model_path]
```

### Debugging GPU Issues
```bash
cd debugging/
python debug_rocm.py  # Comprehensive GPU diagnostics
```

### Model Management
```bash
cd models/
python test_model.py           # Test current models
python integrate_model.py      # Deploy new model
```

## 📂 Directory Overview

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `training/` | Model training scripts | `train_gpu.py` |
| `debugging/` | GPU/ROCm troubleshooting | `debug_rocm.py`, `fix_rocm_rx6800.py` |
| `models/` | Model management | `test_model.py`, `production_text_recognizer.py` |
| `src/core/` | Core OCR logic | `ocr_processor.py`, `text_recognizer.py` |
| `src/api/` | REST API endpoints | `main.py`, `endpoints/` |
| `tools/` | Development utilities | `roi_selector.py`, `snap_to_ink.py` |

## 🔧 Current Status

### ✅ Working
- Base TrOCR model (15% accuracy)
- Synthetic data generation (180 samples)
- CPU training pipeline
- Production OCR integration

### ⚠️ Known Issues
- AMD RX 6800 + ROCm 6.1 segfaults during training
- GPU training unstable with current hardware

### 🎯 Recommended Workflow
1. **Training**: Use CPU training for reliability
2. **Testing**: Validate models with `models/test_model.py`
3. **Deployment**: Use `models/integrate_model.py`
4. **Production**: Leverage `models/production_text_recognizer.py`

## 📚 Documentation

Each directory contains detailed README files:
- `training/README.md` - Training procedures and troubleshooting
- `debugging/README.md` - GPU debugging and diagnostics
- `models/README.md` - Model management and deployment

## 🆘 Quick Help

### GPU Not Working?
```bash
cd debugging/
python debug_rocm.py
```

### Model Performance Poor?
```bash
cd models/
python test_model.py --compare
```

### Need to Retrain?
```bash
cd training/
python train_gpu.py  # Auto-detects CPU fallback
```

## 🔗 Related Files

- `synthetic_data/` - Training data (180 samples)
- `data/templates/` - Form templates
- `requirements*.txt` - Python dependencies
- `.gitignore` - Git exclusions (models, data, cache)