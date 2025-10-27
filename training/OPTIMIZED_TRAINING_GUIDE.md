# 🚀 GPU Training Guide - Production Ready (October 2025)

## 🎯 **Status: FULLY OPERATIONAL**

AMD RX 6800 + ROCm 6.4 + TrOCR training is **completely successful** after BIOS optimizations and system stability improvements.

### ✅ **Proven Results**
- **5 epochs completed**: Zero crashes or system instability
- **Excellent model quality**: 99.97% loss reduction (15.72 → 0.00054)
- **Stable GPU memory**: 1.3GB → 3.8GB progression without issues
- **Fast training**: 6m55s for full fine-tuning cycle
- **Production ready**: High-quality model saved and validated

---

## 🔧 **Environment Setup (Critical)**

### **Required Variables** (Must set before training)
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HIP_LAUNCH_BLOCKING=1
```

### **Activate Environment**
```bash
# Standard venv activation
source venv/bin/activate

# Verify GPU detection
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
```

---

## �️ **Training Mode Options**

### **Option 1: GUI Mode** (Recommended)

**Best for**: Regular training, development, convenience

```bash
# In VS Code terminal or regular desktop terminal
cd /home/isacc/Documents/vs-code/ocr

# Set environment
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HIP_LAUNCH_BLOCKING=1

# Pre-training check (recommended)
python training/verify_stability.py

# Start training
python training/train_gpu.py
```

**Advantages**:
- ✅ Convenient VS Code integration
- ✅ Easy monitoring and debugging
- ✅ Full desktop functionality available
- ✅ Now stable with BIOS optimizations

### **Option 2: TTY Mode** (Maximum Performance)

**Best for**: Maximum GPU dedication, critical training runs, large models

```bash
# Step 1: Switch to TTY
# Press Ctrl+Alt+F2 (or F3-F6)
# Login with credentials

# Step 2: Navigate and setup
cd /home/isacc/Documents/vs-code/ocr
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HIP_LAUNCH_BLOCKING=1

# Step 3: Optional - Stop display manager for max GPU
sudo systemctl stop display-manager

# Step 4: Train
python training/train_gpu.py

# Step 5: Restore desktop when done
sudo systemctl start display-manager
# Press Ctrl+Alt+F1 to return to desktop
```

**Advantages**:
- ✅ 100% GPU dedication
- ✅ Maximum memory available
- ✅ Eliminates Wayland overhead
- ✅ Best for extended training sessions

---

## � **Pre-Training Verification**

**Always run before training** to ensure system stability:

```bash
python training/verify_stability.py
```

**What it checks**:
- ✅ System readiness (GPU, ROCm, environment variables)
- ✅ GPU stress test (memory allocation + computation)
- ✅ TrOCR model stability (loading, inference, training steps)

**Expected output**: All tests pass with green ✅ checkmarks

---

## � **Training Performance Expectations**

### **Optimized Settings Applied**
- **Batch Size**: 6 (aggressive for stable 16GB GPU)
- **Epochs**: 5 (extended for better model quality)
- **Learning Rate**: 3e-5 (optimized for AMD GPU)
- **Monitoring**: Every 3 steps for detailed progress
- **Memory Management**: Conservative allocation with cleanup

### **Expected Metrics**
| Metric | Value | Note |
|--------|-------|------|
| **Training Time** | 6-8 minutes | For 5 epochs |
| **GPU Memory** | 1.3GB → 3.8GB | Stable progression |
| **Training Speed** | 1.7 samples/sec | Efficient processing |
| **Loss Reduction** | 99%+ improvement | Excellent convergence |
| **Final Model Size** | ~1.5GB | Ready for deployment |

---

## ⚠️ **Troubleshooting**

### **If Verification Fails**
```bash
# Check GPU status
rocm-smi

# Verify environment variables
echo $HSA_OVERRIDE_GFX_VERSION
echo $PYTORCH_ALLOC_CONF

# Check system logs
dmesg | grep -i "amd\|rocm\|error"
```

### **If Training Crashes** (Unlikely now)
```bash
# Return to desktop (if in TTY)
sudo systemctl start display-manager
# Press Ctrl+Alt+F1

# Check logs
cat training.log
dmesg | tail -20

# Fallback to CPU training
python training/train_cpu_fallback.py
```

---

## 🎉 **Success Indicators**

**During Training**:
```
🚀 Starting GPU-Accelerated TrOCR Fine-tuning...
🔥 Using AMD GPU: AMD Radeon RX 6800
💾 GPU Memory: 16.0 GB
🎮 Using batch size: 6 (GPU memory: 16.0GB)
💪 Hardware stability verified - using optimized settings
```

**Progress Monitoring**:
```
{'loss': 15.7186, 'epoch': 0.12}  # Starting high
{'loss': 0.7080, 'epoch': 0.62}   # Rapid improvement
{'loss': 0.0027, 'epoch': 1.25}   # Excellent convergence
{'loss': 0.0005, 'epoch': 5.0}    # Final result
```

**Completion**:
```
✅ GPU training completed! Model saved to: ./trocr-gpu-improved
🎯 GPU-trained model result: 'John Smith'
💾 Final GPU memory usage: 3.8GB
```

---

## 🔗 **Next Steps After Training**

```bash
# Test the new model
python models/test_model.py ./trocr-gpu-improved

# Compare with base model
python models/test_model.py --compare

# Deploy to production
python models/integrate_model.py

# Export to ONNX (optional)
python debugging/export_trocr_onnx.py
```

---

**🏆 Your AMD RX 6800 system is now a proven, production-ready GPU training environment!**