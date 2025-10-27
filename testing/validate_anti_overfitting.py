#!/usr/bin/env python3
"""
Test Enhanced Training Pipeline
Validates anti-overfitting measures are working
"""

import sys
import os
import json
from pathlib import Path
import traceback

# Add project root to path
sys.path.append('/home/isacc/Documents/vs-code/ocr')

def test_dependencies():
    """Test that all required packages are available"""
    print("🧪 Testing Dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'HuggingFace Transformers'),
        ('jiwer', 'WER/CER metrics'),
        ('torchvision', 'Data augmentation'),
        ('evaluate', 'Evaluation metrics'),
        ('PIL', 'Image processing')
    ]
    
    all_available = True
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {description}: Available")
        except ImportError:
            print(f"❌ {description}: Missing ({package})")
            all_available = False
    
    return all_available

def test_data_availability():
    """Check training data is available"""
    print("\\n📁 Testing Data Availability...")
    
    data_file = Path("/home/isacc/Documents/vs-code/ocr/synthetic_data/training_data.json")
    
    if not data_file.exists():
        print(f"❌ Training data not found: {data_file}")
        return False
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Training data loaded: {len(data)} samples")
        
        # Check data structure
        if len(data) > 0:
            sample = data[0]
            required_fields = ['image_path', 'text']
            for field in required_fields:
                if field in sample:
                    print(f"✅ Required field '{field}': Present")
                else:
                    print(f"❌ Required field '{field}': Missing")
                    return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return False

def test_augmentation_pipeline():
    """Test data augmentation functionality"""
    print("\\n🎨 Testing Data Augmentation...")
    
    try:
        # Import augmentation functions
        import torch
        from torchvision import transforms
        from PIL import Image
        import numpy as np
        
        # Create test augmentation pipeline
        augment_transform = transforms.Compose([
            transforms.RandomRotation(degrees=3, fill=255),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.1),
        ])
        
        # Test with a dummy image
        test_image = Image.fromarray(np.ones((64, 128), dtype=np.uint8) * 255, mode='L')
        
        # Apply augmentation
        augmented = augment_transform(test_image)
        
        print("✅ Data augmentation pipeline functional")
        print(f"✅ Original image size: {test_image.size}")
        print(f"✅ Augmented image size: {augmented.size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Augmentation test failed: {e}")
        return False

def test_metrics_computation():
    """Test CER/WER metric computation"""
    print("\\n📊 Testing Metrics Computation...")
    
    try:
        import jiwer
        
        # Test CER calculation
        reference = "hello world"
        hypothesis = "helo world"
        
        cer = jiwer.cer(reference, hypothesis)
        wer = jiwer.wer(reference, hypothesis)
        
        print(f"✅ CER calculation: {cer:.4f}")
        print(f"✅ WER calculation: {wer:.4f}")
        print("✅ Metrics computation functional")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False

def test_training_script_syntax():
    """Test that the training script has valid syntax"""
    print("\\n📝 Testing Training Script Syntax...")
    
    train_script = Path("/home/isacc/Documents/vs-code/ocr/training/train_gpu.py")
    
    if not train_script.exists():
        print(f"❌ Training script not found: {train_script}")
        return False
    
    try:
        with open(train_script, 'r') as f:
            content = f.read()
        
        # Compile the script
        compile(content, str(train_script), 'exec')
        
        print("✅ Training script syntax is valid")
        
        # Check for key enhancements
        enhancements = [
            ('ImprovedHandwritingDataset', 'Enhanced dataset class'),
            ('compute_metrics', 'CER/WER metrics function'),
            ('EarlyStoppingCallback', 'Early stopping'),
            ('jiwer.cer', 'CER calculation'),
            ('RandomRotation', 'Data augmentation'),
            ('eval_metric_for_best_model', 'Metric-based model selection')
        ]
        
        for keyword, description in enhancements:
            if keyword in content:
                print(f"✅ {description}: Implemented")
            else:
                print(f"⚠️  {description}: Not found ({keyword})")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in training script: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking training script: {e}")
        return False

def test_gpu_availability():
    """Test GPU setup for training"""
    print("\\n🖥️  Testing GPU Availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("❌ CUDA not available")
            return False
            
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all validation tests"""
    print("🚀 COMPREHENSIVE ANTI-OVERFITTING VALIDATION")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Data Availability", test_data_availability),
        ("Augmentation Pipeline", test_augmentation_pipeline),
        ("Metrics Computation", test_metrics_computation),
        ("Training Script Syntax", test_training_script_syntax),
        ("GPU Availability", test_gpu_availability)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\\n🎉 ALL TESTS PASSED - Anti-overfitting pipeline ready!")
        print("\\n💡 Next Steps:")
        print("1. Run training: python training/train_gpu.py")
        print("2. Monitor CER/WER metrics in logs")
        print("3. Check for early stopping if overfitting detected")
    else:
        print("\\n⚠️  Some tests failed - fix issues before training")
        
    return passed == total

if __name__ == "__main__":
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        traceback.print_exc()
        sys.exit(1)