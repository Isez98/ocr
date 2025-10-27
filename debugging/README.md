# Debugging Scripts

This directory contains essential debugging and diagnostic tools for the OCR project.

## Core Scripts

### `debug_rocm.py`
- **Purpose**: Comprehensive ROCm/GPU debugging and diagnostics
- **Tests**: Basic operations, model loading, inference, memory stress
- **Usage**: `python debug_rocm.py`
- **Output**: Detailed test results and recommendations

### `test_cpu_vs_gpu.py`
- **Purpose**: Compare CPU vs GPU performance and isolate GPU issues
- **Tests**: TrOCR on CPU vs GPU to identify hardware compatibility problems
- **Usage**: `python test_cpu_vs_gpu.py`

### `export_trocr_onnx.py`
- **Purpose**: Export TrOCR to ONNX format for GPU compatibility fallback
- **Features**: Bypasses PyTorch GPU issues, enables alternative deployment
- **Usage**: `python export_trocr_onnx.py`

### `test_onnx_trocr.py`
- **Purpose**: Test ONNX TrOCR inference with various execution providers
- **Features**: ROCm, MIGraphX, and CPU execution provider testing
- **Usage**: `python test_onnx_trocr.py`
- **Usage**: `python fix_rocm_rx6800.py`
- **Status**: ⚠️ Workarounds unsuccessful for RX 6800 + ROCm 6.1

## Common Debug Scenarios

### GPU Detection Issues
```bash
python debug_rocm.py  # Check if GPU is properly detected
```

### Model Loading Problems
- Check transformers library compatibility
- Verify model cache integrity
- Test basic tensor operations

### Training Segfaults
- Known issue with AMD RX 6800 + ROCm 6.1 + TrOCR
- Use debugging scripts to isolate the failure point
- Consider CPU training as fallback

## Environment Variables Tested

- `HSA_OVERRIDE_GFX_VERSION=10.3.0`
- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`
- `HIP_LAUNCH_BLOCKING=1`
- `HIP_FORCE_DEV_KERNARG=1`
- `HSA_ENABLE_SDMA=0`
- `HSA_DISABLE_CACHE=1`

## Debug Output Interpretation

### ✅ Success Indicators
- Basic tensor operations work
- Model loads without errors
- Inference completes successfully
- Memory allocation stable

### ❌ Failure Indicators
- Segmentation faults during inference
- CUDA/ROCm out of memory errors
- Model loading timeouts
- Driver compatibility warnings

## Hardware-Specific Issues

### AMD RX 6800 (RDNA2)
- **ROCm 6.1**: Known segfault with Vision Transformers
- **Workaround**: None found - use CPU training
- **Alternative**: Try different ROCm versions (5.7, 6.0)

### NVIDIA GPUs
- Generally better supported with CUDA
- Use CUDA version of PyTorch for best results

## Getting Help

1. **Run Full Debug**: `python debug_rocm.py`
2. **Check System**: `rocm-smi` and `nvidia-smi`
3. **Verify Environment**: Check Python/PyTorch/ROCm versions
4. **Fallback**: Use CPU training if GPU issues persist