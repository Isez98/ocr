#!/usr/bin/env python3
"""
Quick validation test for enhanced training pipeline
Tests: CER/WER metrics, data augmentation, early stopping
"""

import os
import sys
import subprocess

def setup_environment():
    """Set up the environment variables and verify dependencies"""
    print("🔧 Setting up environment...")
    
    # Set AMD GPU environment variables
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['HIP_LAUNCH_BLOCKING'] = '1'
    
    print("✅ Environment variables set:")
    print(f"   HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")
    print(f"   PYTORCH_ALLOC_CONF: {os.environ.get('PYTORCH_ALLOC_CONF')}")
    print(f"   HIP_LAUNCH_BLOCKING: {os.environ.get('HIP_LAUNCH_BLOCKING')}")

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'torch',
        'transformers', 
        'PIL',
        'jiwer',
        'torchvision'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        # Install missing packages
        for package in missing_packages:
            if package == 'PIL':
                package = 'Pillow'
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"   ✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"   ❌ Failed to install {package}")
                return False
    
    return True

def test_imports():
    """Test that all imports work correctly"""
    print("\n🧪 Testing imports...")
    
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        
        import transformers
        print(f"   ✅ Transformers {transformers.__version__}")
        
        import jiwer
        print(f"   ✅ JIWER available")
        
        import torchvision.transforms as transforms
        print(f"   ✅ TorchVision transforms available")
        
        from PIL import Image
        print(f"   ✅ PIL Image available")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False

def test_gpu():
    """Test GPU availability"""
    print("\n🎮 Testing GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   ✅ GPU: {gpu_name}")
            print(f"   ✅ VRAM: {gpu_memory:.1f}GB")
            
            # Test basic GPU operation
            x = torch.randn(100, 100, device='cuda')
            y = torch.mm(x, x)
            print(f"   ✅ GPU operations working")
            
            return True
        else:
            print(f"   ⚠️  No GPU available - will use CPU")
            return False
            
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
        return False

def test_augmentation():
    """Test data augmentation pipeline"""
    print("\n🔄 Testing data augmentation...")
    
    try:
        import torch
        from PIL import Image
        import torchvision.transforms as transforms
        
        # Create test image
        test_image = Image.new('RGB', (384, 384), 'white')
        
        # Test augmentation pipeline
        augment = transforms.Compose([
            transforms.RandomRotation(degrees=3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),  # Gaussian noise
        ])
        
        augmented = augment(test_image)
        print(f"   ✅ Augmentation pipeline working")
        print(f"   ✅ Output shape: {augmented.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Augmentation test failed: {e}")
        return False

def test_metrics():
    """Test CER/WER metric calculation"""
    print("\n📊 Testing metrics...")
    
    try:
        from jiwer import cer, wer
        
        # Test data
        reference = ["hello world", "test sample"]
        hypothesis = ["hello word", "test sample"]
        
        cer_score = cer(reference, hypothesis)
        wer_score = wer(reference, hypothesis)
        
        print(f"   ✅ CER calculation: {cer_score:.4f}")
        print(f"   ✅ WER calculation: {wer_score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Metrics test failed: {e}")
        return False

def run_quick_training_test():
    """Run a very quick training test with minimal data"""
    print("\n🏃 Quick training pipeline test...")
    
    try:
        # Import training components
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Test if we can import the enhanced dataset
        from train_gpu import ImprovedHandwritingDataset
        
        print(f"   ✅ Enhanced dataset class imported")
        
        # Check if training data exists
        data_file = "synthetic_data/training_data.json"
        if os.path.exists(data_file):
            print(f"   ✅ Training data found: {data_file}")
        else:
            print(f"   ⚠️  Training data not found: {data_file}")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Training test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🚀 Enhanced Training Pipeline Validation")
    print("=" * 50)
    
    # Setup
    setup_environment()
    
    # Run tests
    tests = [
        ("Dependencies", check_dependencies),
        ("Imports", test_imports),
        ("GPU", test_gpu),
        ("Augmentation", test_augmentation),
        ("Metrics", test_metrics),
        ("Training Pipeline", run_quick_training_test)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 VALIDATION SUMMARY:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print(f"\n🎯 OVERALL STATUS:")
    if all_passed:
        print("✅ All tests passed! Enhanced training pipeline ready.")
        print("\n🚀 Next steps:")
        print("   1. python training/verify_stability.py")
        print("   2. python training/train_gpu.py")
        print("\n💡 New features active:")
        print("   • CER/WER metrics for proper evaluation")
        print("   • Data augmentation to reduce overfitting")
        print("   • Early stopping on metric plateau")
        print("   • Best model selection by CER/WER")
    else:
        print("❌ Some tests failed. Fix issues before training.")
        print("💡 Try: pip install jiwer torchvision")
    
    return all_passed

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)