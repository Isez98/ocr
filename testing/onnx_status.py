#!/usr/bin/env python3
"""
ONNX Status Check
Quick status check for ONNX TrOCR functionality
"""

import onnxruntime as ort
from pathlib import Path

def main():
    print("📊 ONNX TrOCR Status Check")
    print("=" * 30)
    
    # Check ONNX Runtime
    print(f"ONNX Runtime: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"Available providers: {len(providers)}")
    for provider in providers:
        status = "✅" if provider == "CPUExecutionProvider" else "⚠️"
        print(f"  {status} {provider}")
    
    # Check model file
    model_path = Path("/home/isacc/Documents/vs-code/ocr/models/trocr.onnx")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model file: {size_mb:.1f} MB")
    else:
        print(f"❌ Model file: Not found")
        return False
    
    # Quick functionality test
    print("\n🧪 Quick Test:")
    try:
        sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        active = sess.get_providers()[0]
        print(f"✅ Model loads with {active}")
        
        # Check inputs/outputs
        inputs = len(sess.get_inputs())
        outputs = len(sess.get_outputs())
        print(f"✅ Model structure: {inputs} inputs, {outputs} outputs")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎯 ONNX TrOCR is ready for use!")
    else:
        print(f"\n💥 ONNX TrOCR needs attention!")
    exit(0 if success else 1)