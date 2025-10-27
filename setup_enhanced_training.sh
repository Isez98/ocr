#!/bin/bash
# Enhanced Training Pipeline Setup and Test Script
# Sets up environment, installs dependencies, and validates the anti-overfitting enhancements

set -e  # Exit on any error

echo "🚀 Enhanced Training Pipeline Setup"
echo "=================================="

# Function to print colored output
print_status() {
    echo -e "\033[1;32m[INFO]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARN]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Step 1: Activate virtual environment
print_status "Step 1: Activating virtual environment..."

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    print_status "✅ Virtual environment activated"
elif [ -f "/storage/.venv/ocr_stable_py311/bin/activate" ]; then
    source /storage/.venv/ocr_stable_py311/bin/activate
    print_status "✅ Stable virtual environment activated"
else
    print_warning "No virtual environment found. Creating new one..."
    python -m venv venv
    source venv/bin/activate
    print_status "✅ New virtual environment created and activated"
fi

# Step 2: Set AMD GPU environment variables
print_status "Step 2: Setting AMD GPU environment variables..."

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HIP_LAUNCH_BLOCKING=1

print_status "✅ Environment variables set:"
print_status "   HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
print_status "   PYTORCH_ALLOC_CONF: $PYTORCH_ALLOC_CONF" 
print_status "   HIP_LAUNCH_BLOCKING: $HIP_LAUNCH_BLOCKING"

# Step 3: Update pip and install core dependencies
print_status "Step 3: Installing/updating dependencies..."

pip install --upgrade pip

# Install new required packages for anti-overfitting features
print_status "Installing jiwer for CER/WER metrics..."
pip install jiwer

print_status "Installing torchvision for data augmentation..."
pip install torchvision

print_status "Ensuring transformers and torch are up to date..."
pip install --upgrade torch transformers

print_status "✅ Dependencies installed"

# Step 4: Verify basic functionality
print_status "Step 4: Running basic verification..."

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'GPU Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')

import transformers
print(f'Transformers: {transformers.__version__}')

import jiwer
print('JIWER: Available for CER/WER metrics')

import torchvision
print(f'TorchVision: {torchvision.__version__}')

print('✅ All core dependencies working')
"

# Step 5: Run enhanced training validation
print_status "Step 5: Running enhanced training validation..."

python training/test_enhanced_training.py

# Step 6: Provide usage instructions
print_status "Step 6: Setup complete! Usage instructions:"

cat << 'EOF'

🎯 ENHANCED TRAINING READY!

Quick Start:
============

# 1. Verify system stability (recommended)
python training/verify_stability.py

# 2. Run enhanced training with anti-overfitting features
python training/train_gpu.py

# 3. Test the improved model
python models/test_model.py ./trocr-gpu-improved

New Features Active:
===================

✅ CER/WER Metrics - Real evaluation beyond loss
✅ Data Augmentation - Rotation, perspective, brightness, noise
✅ Early Stopping - Prevents overfitting on CER/WER plateau  
✅ Best Model Selection - Saves best checkpoint by metrics

Expected Improvements:
=====================

🎯 Better generalization to unseen handwriting styles
📊 Meaningful metrics (CER/WER) instead of just loss
🛡️ Protection against overfitting on small dataset
🏆 Higher quality models for real-world deployment

Monitoring:
===========

Watch for:
- CER/WER metrics in training logs
- Early stopping when metrics plateau
- Improved validation performance
- Better real-world recognition

EOF

print_status "🎉 Setup complete! Enhanced training pipeline ready to use."