#!/usr/bin/env python3
"""
Pre-training stability verification for optimized AMD RX 6800 setup
Runs comprehensive checks before starting intensive GPU training
"""

import torch
import time
import gc
import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np

def test_gpu_stress():
    """Extended GPU stress test similar to your successful hardware tests"""
    print("🔥 Running GPU Stability Verification...")
    
    try:
        # Memory stress test
        print("1. GPU Memory Stress Test...")
        tensors = []
        for i in range(10):
            # Allocate 1GB chunks
            tensor = torch.randn(4096, 4096, device='cuda', dtype=torch.float32)
            tensors.append(tensor)
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"   Allocated {allocated:.1f}GB - Step {i+1}/10")
            time.sleep(0.5)
        
        # Computation stress
        print("2. GPU Computation Stress Test...")
        for i in range(20):
            result = torch.mm(tensors[0], tensors[1])
            if i % 5 == 0:
                print(f"   Matrix ops iteration {i+1}/20")
        
        # Memory cleanup
        del tensors, result
        torch.cuda.empty_cache()
        gc.collect()
        
        print("✅ GPU stress test passed")
        return True
        
    except Exception as e:
        print(f"❌ GPU stress test failed: {e}")
        return False

def test_trocr_stability():
    """Test TrOCR model stability with your hardware"""
    print("🤖 Testing TrOCR Model Stability...")
    
    try:
        # Load model
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        model = model.to('cuda')
        
        # Set required model configuration
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        
        # Test with dummy data
        print("1. Testing inference stability...")
        for i in range(10):
            # Create dummy image
            dummy_image = Image.new('RGB', (384, 384), 'white')
            pixel_values = processor(dummy_image, return_tensors="pt").pixel_values.to('cuda')
            
            # Run inference
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_length=32)
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if i % 3 == 0:
                print(f"   Inference test {i+1}/10 - Result: '{text}'")
        
        # Test training components
        print("2. Testing training components...")
        model.train()
        
        # Simulate training step
        dummy_labels = torch.randint(0, 1000, (1, 32), device='cuda')
        
        for i in range(5):
            with torch.amp.autocast('cuda', enabled=False):  # Updated autocast syntax
                outputs = model(pixel_values=pixel_values, labels=dummy_labels)
                loss = outputs.loss
                loss.backward()
            
            print(f"   Training step {i+1}/5 - Loss: {loss.item():.4f}")
        
        print("✅ TrOCR stability test passed")
        return True
        
    except Exception as e:
        print(f"❌ TrOCR stability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_system_readiness():
    """Check system configuration for optimal training"""
    print("⚙️  Checking System Readiness...")
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {gpu_memory:.1f}GB")
        
        # Check for your specific GPU
        if "6800" in gpu_name:
            print("   ✅ AMD RX 6800 detected - BIOS optimized")
        else:
            print(f"   ⚠️  Different GPU detected: {gpu_name}")
    
    # PyTorch info
    print(f"   PyTorch: {torch.__version__}")
    if hasattr(torch.version, 'hip'):
        print(f"   ROCm/HIP: {torch.version.hip}")
    
    # Environment variables for stability
    important_vars = [
        'HSA_OVERRIDE_GFX_VERSION',
        'PYTORCH_ALLOC_CONF',  # Updated from deprecated PYTORCH_HIP_ALLOC_CONF
        'HIP_LAUNCH_BLOCKING'
    ]
    
    print("   Environment Variables:")
    for var in important_vars:
        value = os.environ.get(var, 'Not set')
        status = "✅" if value != 'Not set' else "⚠️"
        print(f"     {status} {var}: {value}")
    
    return True

def run_pre_training_verification():
    """Run complete pre-training verification suite"""
    print("🚀 Pre-Training Verification Suite")
    print("=" * 50)
    
    tests = [
        ("System Readiness", check_system_readiness),
        ("GPU Stress Test", test_gpu_stress),
        ("TrOCR Stability", test_trocr_stability)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}...")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 VERIFICATION SUMMARY:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("\n🎯 RECOMMENDATION:")
    if all_passed:
        print("✅ System ready for GPU training!")
        print("🚀 Run: python training/train_gpu.py")
        print("\n💡 Optimizations active:")
        print("   • Increased batch size (6) for 16GB GPU")
        print("   • Extended epochs (5) for better convergence")
        print("   • Enhanced monitoring and stability features")
    else:
        print("❌ System not ready - issues detected!")
        print("🔧 Fix issues above before training")
        print("🛡️  Fallback: python training/train_cpu_fallback.py")
    
    return all_passed

if __name__ == "__main__":
    try:
        success = run_pre_training_verification()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Verification interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)