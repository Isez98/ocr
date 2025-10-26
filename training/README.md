# Training Scripts

This directory contains all model training scripts for the OCR project.

## Scripts Overview

### `train_gpu.py`
- **Purpose**: GPU-accelerated training using ROCm/CUDA
- **Status**: ⚠️ Known segfault issues with AMD RX 6800 + ROCm 6.1
- **Usage**: `python train_gpu.py`
- **Requirements**: ROCm-enabled PyTorch, GPU with sufficient VRAM

## Training Data

- **Location**: `../synthetic_data/`
- **Format**: JSON with image paths and expected text
- **Samples**: 180 improved synthetic handwriting samples

## Model Output

- **Default Output**: `./trocr-handwritten-finetuned/`
- **Base Model**: `microsoft/trocr-base-handwritten`
- **Fine-tuned For**: Handwritten text recognition

## Common Issues

### AMD RX 6800 + ROCm Segfaults
- **Problem**: TrOCR inference causes segmentation faults
- **Root Cause**: RDNA2 + ROCm 6.1 incompatibility with Vision Transformers
- **Solution**: Use CPU training or different GPU/ROCm version

### Memory Issues
- **Symptoms**: CUDA out of memory errors
- **Solutions**: 
  - Reduce batch size
  - Use gradient accumulation
  - Enable mixed precision training

## Training Tips

1. **Start Small**: Begin with small datasets and few epochs
2. **Monitor Progress**: Check loss curves and validation metrics
3. **Save Frequently**: Use checkpoint saving to avoid losing progress
4. **Test Early**: Validate model performance with `../models/test_model.py`

## Next Steps After Training

1. **Test Model**: `python ../models/test_model.py`
2. **Compare Performance**: Use comparison features in test script
3. **Deploy Model**: `python ../models/integrate_model.py`
4. **Production Use**: Configure `../models/production_text_recognizer.py`