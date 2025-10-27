#!/usr/bin/env python3
"""
Export TrOCR to ONNX for GPU acceleration via MIGraphX/ROCm EP
This bypasses PyTorch attention kernels that cause segfaults
"""

from transformers import VisionEncoderDecoderModel, AutoProcessor
import torch
import os

def export_trocr_onnx():
    print("📦 TrOCR ONNX Export")
    print("=" * 40)
    
    print(f"PyTorch: {torch.__version__}")
    
    # Use CPU for export to avoid segfaults
    device = "cpu"
    
    try:
        print("\n1. Loading TrOCR model...")
        mname = "microsoft/trocr-base-stage1"
        model = VisionEncoderDecoderModel.from_pretrained(mname).eval()
        proc = AutoProcessor.from_pretrained(mname)
        print(f"   ✅ Model loaded on {device}")
        
        print("\n2. Creating dummy input...")
        dummy_pixel_values = torch.randn(1, 3, 384, 384)  # TrOCR default input size
        dummy_decoder_input_ids = torch.tensor([[2]])  # Start token for decoder
        print(f"   ✅ Dummy inputs: pixel_values={dummy_pixel_values.shape}, decoder_input_ids={dummy_decoder_input_ids.shape}")
        
        print("\n3. Exporting to ONNX...")
        output_path = "models/trocr.onnx"
        os.makedirs("models", exist_ok=True)
        
        torch.onnx.export(
            model, 
            (dummy_pixel_values, dummy_decoder_input_ids), 
            output_path,
            input_names=["pixel_values", "decoder_input_ids"], 
            output_names=["logits"],
            dynamic_axes={
                "pixel_values": {0: "batch"}, 
                "decoder_input_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"}
            },
            opset_version=17, 
            do_constant_folding=True,
            verbose=False
        )
        
        print(f"   ✅ Exported to {output_path}")
        
        # Check file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"   File size: {file_size:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = export_trocr_onnx()
    if success:
        print("\n🎉 ONNX export successful!")
        print("📝 Next step: Install ONNX Runtime with ROCm support")
        print("   pip install onnxruntime-rocm")
    else:
        print("\n💥 ONNX export failed")
    exit(0 if success else 1)