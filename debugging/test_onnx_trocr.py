#!/usr/bin/env python3
"""
Test ONNX TrOCR with ROCm/MIGraphX execution providers
This should work even if PyTorch segfaults
"""

import numpy as np

def test_onnx_trocr():
    print("🚀 ONNX TrOCR GPU Test")
    print("=" * 40)
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ ONNX Runtime not installed")
        print("   Install with: pip install onnxruntime-rocm")
        return False
        
    try:
        print(f"ONNX Runtime: {ort.__version__}")
        
        # Check available providers
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        
        print("\n1. Loading ONNX model...")
        model_path = "models/trocr.onnx"
        
        # Try to use ROCm/MIGraphX providers
        target_providers = [
            ("MIGraphXExecutionProvider", {}),
            ("ROCMExecutionProvider", {}), 
            "CPUExecutionProvider"
        ]
        
        # Filter to only available providers
        available_providers = []
        for provider in target_providers:
            if isinstance(provider, tuple):
                provider_name = provider[0]
            else:
                provider_name = provider
                
            if provider_name in providers:
                available_providers.append(provider)
                print(f"   ✅ Using {provider_name}")
            else:
                print(f"   ❌ {provider_name} not available")
        
        sess = ort.InferenceSession(model_path, providers=available_providers)
        print(f"   ✅ Session created with providers: {sess.get_providers()}")
        
        print("\n2. Creating test input...")
        # Create a test tensor like TrOCR processor would output
        pixel_values = np.random.randn(1, 3, 384, 384).astype("float32")
        decoder_input_ids = np.array([[2]], dtype=np.int64)  # Start token
        print(f"   ✅ Input shapes: pixel_values={pixel_values.shape}, decoder_input_ids={decoder_input_ids.shape}")
        
        print("\n3. Running inference...")
        y = sess.run(None, {
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids
        })
        print(f"   ✅ Inference successful!")
        print(f"   Output shapes: {[t.shape for t in y]}")
        
        # Check which provider was actually used
        actual_provider = sess.get_providers()[0]
        if "ROCM" in actual_provider or "MIGraphX" in actual_provider:
            print(f"   🎉 Running on GPU via {actual_provider}")
        else:
            print(f"   ⚠️  Running on CPU via {actual_provider}")
        
        return True
        
    except ImportError:
        print("❌ ONNX Runtime not installed")
        print("   Install with: pip install onnxruntime-rocm")
        return False
    except FileNotFoundError:
        print("❌ ONNX model not found")
        print("   Run export_trocr_onnx.py first")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_onnx_trocr()
    if success:
        print("\n🎉 ONNX TrOCR test passed!")
    else:
        print("\n💥 ONNX TrOCR test failed")
    exit(0 if success else 1)