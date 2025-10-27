# Training Scripts

This directory contains training scripts for the OCR project.

## Available Scripts

- `train_gpu.py` - GPU-accelerated training (primary, works on NVIDIA GPUs)
- `train_cpu_fallback.py` - CPU fallback training (recommended for AMD RX 6800 due to ROCm compatibility issues)

## Usage

For AMD RX 6800 users experiencing GPU segmentation faults:
```bash
source /storage/venv_profile.sh && activate_ocr
python training/train_cpu_fallback.py
```

For NVIDIA GPU users:
```bash
python training/train_gpu.py
```