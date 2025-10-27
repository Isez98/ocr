#!/usr/bin/env python3
"""
CPU Training Script for AMD RX 6800 ROCm Compatibility Issues

This script runs TrOCR training on CPU since the AMD RX 6800 + ROCm 6.1
combination causes segmentation faults with Vision Transformers.

While slower than GPU, this allows for functional training and testing.
"""

import torch
import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

def train_cpu_trocr():
    print("🖥️  CPU TrOCR Training")
    print("=" * 50)
    print("⚠️  Running on CPU due to AMD RX 6800 + ROCm segfault issues")
    print("🔧 This is slower but functional for training/testing")
    print()
    
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    
    # Force CPU usage
    device = torch.device("cpu")
    print(f"Device: {device}")
    
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from transformers import Trainer, TrainingArguments
        from transformers import default_data_collator
        
        print("\n📥 Loading TrOCR model...")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        
        # Keep model on CPU
        model = model.to(device)
        print(f"✅ Model loaded on {device}")
        
        # Test basic functionality
        print("\n🧪 Testing model inference...")
        dummy_image = torch.randn(1, 3, 384, 384)
        
        with torch.no_grad():
            outputs = model.generate(dummy_image, max_length=20)
            print(f"✅ Generation works: {outputs.shape}")
        
        print("\n🎯 CPU training setup ready!")
        print("📝 To start training:")
        print("   1. Prepare your training data")
        print("   2. Update training arguments for CPU (lower batch size)")
        print("   3. Run training with device='cpu'")
        print("\n💡 Consider using smaller models or quantization for faster CPU training")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = train_cpu_trocr()
    exit(0 if success else 1)