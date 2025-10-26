#!/usr/bin/env python3
"""
Comprehensive ROCm debugging and troubleshooting script
"""

import torch
import gc
import os
import sys
import traceback
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np

def test_basic_operations():
    """Test basic ROCm operations"""
    print("🔍 Testing Basic ROCm Operations...")
    
    try:
        # Basic tensor operations
        print("1. Creating tensors...")
        x = torch.randn(100, 100, device='cuda')
        y = torch.randn(100, 100, device='cuda')
        
        print("2. Matrix multiplication...")
        z = torch.mm(x, y)
        
        print("3. Memory allocation test...")
        large_tensor = torch.randn(1000, 1000, device='cuda')
        del large_tensor
        
        print("✅ Basic operations successful")
        return True
        
    except Exception as e:
        print(f"❌ Basic operations failed: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test TrOCR model loading on GPU"""
    print("\n🔍 Testing TrOCR Model Loading...")
    
    try:
        print("1. Loading processor...")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        
        print("2. Loading model...")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        
        print("3. Moving to GPU...")
        model = model.to('cuda')
        
        print("4. Setting model config...")
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        
        print("✅ Model loading successful")
        return model, processor
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        traceback.print_exc()
        return None, None

def test_model_inference(model, processor):
    """Test simple model inference"""
    print("\n🔍 Testing Model Inference...")
    
    try:
        print("1. Creating test image...")
        # Create simple test image
        test_image = Image.new('RGB', (200, 80), 'white')
        
        print("2. Processing image...")
        pixel_values = processor(test_image, return_tensors="pt").pixel_values.to('cuda')
        
        print("3. Running inference...")
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=16)
        
        print("4. Decoding result...")
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"   Result: '{text}'")
        
        print("✅ Inference successful")
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        traceback.print_exc()
        return False

def test_training_setup():
    """Test training components"""
    print("\n🔍 Testing Training Setup...")
    
    try:
        from torch.utils.data import DataLoader, Dataset
        from transformers import TrainingArguments
        
        print("1. Creating dummy dataset...")
        class DummyDataset(Dataset):
            def __init__(self):
                self.data = [{'pixel_values': torch.randn(3, 384, 384), 
                             'labels': torch.randint(0, 1000, (16,))} for _ in range(4)]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        print("2. Creating dataloader...")
        dataset = DummyDataset()
        dataloader = DataLoader(dataset, batch_size=1)
        
        print("3. Testing batch processing...")
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to('cuda')
            labels = batch['labels'].to('cuda')
            print(f"   Batch shape: {pixel_values.shape}")
            break
        
        print("✅ Training setup successful")
        return True
        
    except Exception as e:
        print(f"❌ Training setup failed: {e}")
        traceback.print_exc()
        return False

def test_memory_stress():
    """Test memory allocation patterns that might cause segfaults"""
    print("\n🔍 Testing Memory Stress Patterns...")
    
    try:
        print("1. Gradual memory allocation...")
        tensors = []
        for i in range(5):
            tensor = torch.randn(512, 512, device='cuda')
            tensors.append(tensor)
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"   Step {i+1}: {allocated:.2f} GB allocated")
        
        print("2. Memory cleanup...")
        del tensors
        gc.collect()
        torch.cuda.empty_cache()
        
        print("3. Large allocation test...")
        # Try allocating 2GB
        large_tensor = torch.randn(16384, 16384, device='cuda')  # ~1GB
        print(f"   Large tensor shape: {large_tensor.shape}")
        del large_tensor
        
        print("✅ Memory stress test successful")
        return True
        
    except Exception as e:
        print(f"❌ Memory stress test failed: {e}")
        traceback.print_exc()
        return False

def check_environment():
    """Check environment variables and settings"""
    print("\n🔍 Checking Environment...")
    
    rocm_vars = [
        'HIP_VISIBLE_DEVICES',
        'ROCR_VISIBLE_DEVICES', 
        'HSA_OVERRIDE_GFX_VERSION',
        'PYTORCH_HIP_ALLOC_CONF',
        'HIP_FORCE_DEV_KERNARG'
    ]
    
    print("ROCm Environment Variables:")
    for var in rocm_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")
    
    print(f"\nPython: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    if hasattr(torch.version, 'hip'):
        print(f"HIP: {torch.version.hip}")
    
    return True

def run_comprehensive_debug():
    """Run all debugging tests"""
    print("🚀 Starting Comprehensive ROCm Debugging...\n")
    
    results = {}
    
    # Check environment
    results['environment'] = check_environment()
    
    # Test basic operations
    results['basic_ops'] = test_basic_operations()
    
    if not results['basic_ops']:
        print("\n❌ Basic operations failed - ROCm installation issue")
        return results
    
    # Test model loading
    model, processor = test_model_loading()
    results['model_loading'] = (model is not None)
    
    if model is None:
        print("\n❌ Model loading failed - transformers/ROCm compatibility issue")
        return results
    
    # Test inference
    results['inference'] = test_model_inference(model, processor)
    
    # Test memory stress
    results['memory_stress'] = test_memory_stress()
    
    # Test training setup
    results['training_setup'] = test_training_setup()
    
    # Summary
    print("\n" + "="*60)
    print("🏁 DEBUG SUMMARY:")
    print("="*60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test.upper():20} {status}")
    
    # Recommendations
    print("\n🔧 RECOMMENDATIONS:")
    
    if not results['basic_ops']:
        print("• ROCm driver/runtime issue - reinstall ROCm")
    elif not results['model_loading']:
        print("• Transformers library incompatibility - try different versions")
    elif not results['inference']:
        print("• Model execution issue - likely memory/driver problem")
    elif not results['memory_stress']:
        print("• Memory allocation issue - reduce batch size significantly")
    elif not results['training_setup']:
        print("• Training framework issue - use CPU training instead")
    else:
        print("• All tests passed - training segfault is likely:")
        print("  - ROCm/PyTorch version mismatch")
        print("  - Specific training loop issue")
        print("  - Try environment variables:")
        print("    export HSA_OVERRIDE_GFX_VERSION=10.3.0")
        print("    export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True")
    
    return results

if __name__ == "__main__":
    try:
        results = run_comprehensive_debug()
        
        print(f"\n📊 Overall Success Rate: {sum(results.values())}/{len(results)} tests passed")
        
        if sum(results.values()) < len(results):
            print("\n💡 Consider using CPU training for reliability")
        
    except Exception as e:
        print(f"💥 Critical debugging error: {e}")
        traceback.print_exc()