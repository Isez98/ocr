# GPU Bug Report: AMD RX 6800 + ROCm 6.1 + TrOCR Segmentation Fault

## 🚨 Bug Summary
**Segmentation fault when running TrOCR Vision Transformer on AMD RX 6800 with ROCm 6.1**

**Severity**: Critical - Complete GPU functionality loss for Vision Transformers
**Status**: Unresolved - Hardware/driver compatibility issue
**Reproducibility**: 100% consistent

## 🖥️ System Configuration

### Hardware
- **GPU**: AMD Radeon RX 6800 (16GB VRAM, RDNA2 architecture)
- **CPU**: [System specs not specified]
- **RAM**: [System specs not specified]

### Software Stack
- **OS**: Linux (distribution not specified)
- **ROCm Version**: 6.1.40091-a8dbc0c19
- **Python**: 3.11.14
- **PyTorch**: 2.6.0+rocm6.1 (official ROCm build)
- **Transformers**: 4.46.3 (stable release)
- **HIP Runtime**: 6.1.40091-a8dbc0c19

## 🔍 Problem Description

### Specific Issue
TrOCR (Vision Transformer for handwriting recognition) consistently crashes with segmentation faults when:
1. Model is moved to GPU (`model.to("cuda:0")`)
2. Vision Transformer encoder processes input tensors
3. During forward pass of attention mechanisms

### Error Details
```bash
Segmentation fault (core dumped)
Exit Code: 139
```

## 📋 Reproduction Steps

### Minimal Reproduction Case
```python
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Move to GPU - this succeeds
model = model.to("cuda:0")

# Create dummy input
dummy_image = torch.randn(1, 3, 384, 384).to("cuda:0")

# This line triggers segmentation fault
with torch.no_grad():
    encoder_outputs = model.encoder(dummy_image)  # SEGFAULT HERE
```

### Full Reproduction Script
```bash
# Activate environment
source /storage/venv_profile.sh && activate_ocr

# Run test
python debugging/test_cpu_vs_gpu.py
```

## ✅ What Works

### CPU Execution
- ✅ Model loads successfully
- ✅ Inference runs without issues
- ✅ Generation produces expected outputs
- ✅ All Transformer operations function normally

### Basic GPU Operations
- ✅ `torch.cuda.is_available()` returns `True`
- ✅ GPU device detection works
- ✅ Simple tensor operations on GPU succeed
- ✅ Memory allocation to GPU works
- ✅ Model transfer to GPU completes without error

## ❌ What Fails

### GPU Vision Transformer Operations
- ❌ Vision Transformer encoder forward pass
- ❌ TrOCR model generation on GPU
- ❌ Any Vision Transformer attention computations
- ❌ Multi-head attention mechanisms in ViT

### Failure Point Analysis
```
1. Model Loading: ✅ SUCCESS
2. GPU Transfer: ✅ SUCCESS  
3. Input Creation: ✅ SUCCESS
4. Encoder Forward Pass: ❌ SEGFAULT (Critical failure point)
```

## 🧪 Testing Results

### CPU vs GPU Comparison
| Operation | CPU | GPU |
|-----------|-----|-----|
| Model Loading | ✅ Works | ✅ Works |
| Memory Transfer | N/A | ✅ Works |
| Encoder Forward | ✅ Works | ❌ Segfault |
| Generation | ✅ Works | ❌ Segfault |

### Tested Workarounds (All Failed)
1. **SDPA Backend Control**: Switched to math-only attention backend
2. **Memory Management**: Adjusted memory allocation patterns  
3. **Model Precision**: Tested different float precisions
4. **Batch Size**: Reduced to single sample
5. **Python Versions**: Tested 3.11 (stable) vs 3.13 (dev)
6. **Package Versions**: Downgraded to stable releases

## 🔬 Technical Analysis

### Root Cause Assessment
The segmentation fault occurs specifically in Vision Transformer attention mechanisms when executed on AMD RX 6800 with ROCm 6.1. This suggests:

1. **Hardware-specific issue**: RDNA2 architecture incompatibility with certain attention operations
2. **ROCm driver bug**: Potential issue in ROCm 6.1 memory management for complex tensor operations
3. **PyTorch+ROCm integration**: Possible bug in PyTorch's ROCm backend for Vision Transformers

### Call Stack Analysis
```
Segfault occurs in:
model.encoder() -> ViTModel.forward() -> attention_layer.forward() -> [SEGFAULT]
```

## 🐛 Related Issues

### Known AMD ROCm Issues
- Vision Transformer models have historically had compatibility issues with AMD GPUs
- ROCm 6.1 is relatively new and may have unresolved bugs with transformer architectures
- RDNA2 architecture has known issues with certain PyTorch operations

### Community Reports
- Similar issues reported with other Vision Transformer models on AMD GPUs
- ROCm GitHub repository has related segfault reports
- PyTorch ROCm support forums discuss similar compatibility problems

## 💡 Workarounds

### Current Solution: CPU Fallback
```bash
# Use CPU training instead
python training/train_cpu_fallback.py
```

**Status**: ✅ Functional
**Performance**: Slower but stable

### Alternative Approaches
1. **Cloud GPU**: Use NVIDIA-based cloud instances
2. **Different ROCm Version**: Try ROCm 5.7 or 6.0
3. **Alternative Models**: Non-Vision Transformer OCR approaches
4. **ONNX Runtime**: Convert model to ONNX with ROCm provider

## 📊 Impact Assessment

### Development Impact
- **Immediate**: Can continue development using CPU
- **Performance**: Training/inference significantly slower on CPU
- **Functionality**: No feature loss, only performance degradation

### Production Impact
- **Deployment**: Must use CPU or alternative GPU solutions
- **Scalability**: Limited by CPU performance constraints
- **Cost**: Higher compute costs for CPU-only inference

## 🎯 Recommended Actions

### Short Term
1. Continue development using CPU fallback
2. Monitor ROCm updates for fixes
3. Consider cloud-based GPU training for production

### Long Term
1. Evaluate alternative GPU hardware (NVIDIA RTX series)
2. Explore non-Vision Transformer OCR models
3. Consider hybrid CPU/cloud deployment strategy

## 📞 Support Channels

### Report Locations
- **AMD ROCm**: https://github.com/RadeonOpenCompute/ROCm/issues
- **PyTorch**: https://github.com/pytorch/pytorch/issues  
- **Transformers**: https://github.com/huggingface/transformers/issues

### Bug Report Template
```
Title: Segmentation fault with Vision Transformer on AMD RX 6800 + ROCm 6.1

Environment:
- GPU: AMD Radeon RX 6800
- ROCm: 6.1.40091-a8dbc0c19
- PyTorch: 2.6.0+rocm6.1
- Transformers: 4.46.3

Issue: TrOCR Vision Transformer segfaults during encoder forward pass on GPU
Reproducible: 100% consistent
Workaround: CPU execution works fine
```

---
**Document Status**: Complete analysis of AMD RX 6800 + ROCm 6.1 compatibility issues with Vision Transformers
**Last Updated**: October 26, 2025