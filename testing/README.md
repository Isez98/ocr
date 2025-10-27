# Testing Scripts

Comprehensive testing suite for OCR functionality and performance validation.

## 🧪 **Test Scripts**

### `onnx_status.py`
- **Purpose**: Quick health check for ONNX Runtime and model availability
- **Usage**: `python testing/onnx_status.py`
- **Output**: Provider status, model file validation, basic functionality test

### `quick_onnx_test.py`  
- **Purpose**: Fast ONNX inference test with real sample images
- **Usage**: `python testing/quick_onnx_test.py`
- **Features**: Performance timing, real image processing, decoded text output

### `test_onnx_comprehensive.py`
- **Purpose**: Full ONNX provider testing and benchmarking suite
- **Usage**: `python testing/test_onnx_comprehensive.py`
- **Features**: Multi-provider testing, performance comparison, detailed diagnostics

## 📊 **Expected Results**

### **Working Configuration**
- ✅ **CPUExecutionProvider**: Fully functional, ~0.36-0.43s per inference
- ⚠️ **MIGraphXExecutionProvider**: Falls back to CPU (missing libraries)
- ❌ **ROCMExecutionProvider**: Kernel compatibility issues with RX 6800

### **Performance Benchmarks**
- **Model Loading**: ~1-2 seconds
- **GPU Memory**: 1.29 GB usage
- **Inference Speed**: 0.09-0.68s per image
- **Model Size**: 1.47 GB ONNX file

## 🎯 **Testing Workflow**

1. **Quick Check**: `python testing/onnx_status.py`
2. **Real Test**: `python testing/quick_onnx_test.py` 
3. **Full Benchmark**: `python testing/test_onnx_comprehensive.py`

All tests should pass with the current PyTorch 2.9.0+rocm6.4 configuration.