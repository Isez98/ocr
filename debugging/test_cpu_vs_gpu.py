#!/usr/bin/env python3
"""
Test TrOCR on CPU vs GPU to isolate AMD/ROCm issue
"""

import torch
import sys

def test_cpu_vs_gpu():
    print("🔬 CPU vs GPU TrOCR Test")
    print("=" * 40)
    
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    
    print("\n1. Loading model and processor...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    print("   ✅ Model loaded")
    
    # Test CPU first
    print("\n2. Testing on CPU...")
    try:
        model_cpu = model.to("cpu")
        dummy_image = torch.randn(1, 3, 384, 384)
        
        with torch.no_grad():
            encoder_outputs = model_cpu.encoder(dummy_image)
            print(f"   ✅ CPU encoder works: {encoder_outputs.last_hidden_state.shape}")
            
            # Try generation on CPU
            outputs = model_cpu.generate(dummy_image, max_length=10)
            print(f"   ✅ CPU generation works: {outputs.shape}")
            
    except Exception as e:
        print(f"   ❌ CPU failed: {e}")
        return False
    
    # Test GPU
    if torch.cuda.is_available():
        print(f"\n3. Testing on GPU ({torch.cuda.get_device_name()})...")
        try:
            model_gpu = model.to("cuda:0")
            dummy_image_gpu = torch.randn(1, 3, 384, 384).to("cuda:0")
            
            print("   Testing encoder...")
            with torch.no_grad():
                encoder_outputs = model_gpu.encoder(dummy_image_gpu)
                print(f"   ✅ GPU encoder works: {encoder_outputs.last_hidden_state.shape}")
                
                print("   Testing generation...")
                outputs = model_gpu.generate(dummy_image_gpu, max_length=10)
                print(f"   ✅ GPU generation works: {outputs.shape}")
                
        except Exception as e:
            print(f"   ❌ GPU failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\n3. No GPU available")
    
    print("\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    success = test_cpu_vs_gpu()
    exit(0 if success else 1)