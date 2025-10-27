#!/usr/bin/env python3
"""
Quick ONNX TrOCR Test Runner
Simple script to quickly test ONNX TrOCR functionality
"""

import time
import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor

def main():
    print("🚀 Quick ONNX TrOCR Test")
    print("=" * 40)
    
    # Check ONNX Runtime
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"Providers: {ort.get_available_providers()}")
    
    try:
        # Load model with best available provider
        print("\n1. Loading ONNX model...")
        available_providers = ort.get_available_providers()
        
        # Prefer MIGraphX (falls back to CPU) over ROCm (fails) over CPU
        if 'MIGraphXExecutionProvider' in available_providers:
            providers = ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
            provider_name = "MIGraphX (CPU fallback)"
        elif 'CPUExecutionProvider' in available_providers:
            providers = ['CPUExecutionProvider']  
            provider_name = "CPU"
        else:
            providers = None
            provider_name = "Auto"
            
        sess = ort.InferenceSession('/home/isacc/Documents/vs-code/ocr/models/trocr.onnx', 
                                   providers=providers)
        
        active_providers = sess.get_providers()
        print(f"   ✅ Model loaded with {provider_name}")
        print(f"   Active providers: {active_providers}")
        
        # Load processor
        print("\n2. Loading TrOCR processor...")
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        print("   ✅ Processor loaded")
        
        # Test with real sample image
        sample_path = "/home/isacc/Documents/vs-code/ocr/samples/scan_form_1.jpg"
        print(f"\n3. Testing with sample image: {sample_path}")
        
        try:
            image = Image.open(sample_path).convert('RGB')
            print(f"   Image size: {image.size}")
            
            # Process image
            pixel_values = processor(image, return_tensors='pt').pixel_values.numpy()
            decoder_input_ids = np.array([[processor.tokenizer.bos_token_id]], dtype=np.int64)
            
            # Run inference
            start_time = time.time()
            ort_inputs = {
                'pixel_values': pixel_values,
                'decoder_input_ids': decoder_input_ids
            }
            ort_outputs = sess.run(None, ort_inputs)
            inference_time = time.time() - start_time
            
            # Decode result
            logits = ort_outputs[0]
            predicted_ids = np.argmax(logits, axis=-1)
            text = processor.decode(predicted_ids[0], skip_special_tokens=True)
            
            print(f"   ✅ Inference completed ({inference_time:.3f}s)")
            print(f"   Recognized text: \"{text}\"")
            print(f"   Output shape: {logits.shape}")
            
        except FileNotFoundError:
            print("   ⚠️  Sample image not found, using blank test image")
            
            # Fallback to blank image test
            test_image = Image.new('RGB', (384, 384), color='white')
            pixel_values = processor(test_image, return_tensors='pt').pixel_values.numpy()
            decoder_input_ids = np.array([[processor.tokenizer.bos_token_id]], dtype=np.int64)
            
            start_time = time.time()
            ort_inputs = {
                'pixel_values': pixel_values,
                'decoder_input_ids': decoder_input_ids
            }
            ort_outputs = sess.run(None, ort_inputs)
            inference_time = time.time() - start_time
            
            logits = ort_outputs[0]
            predicted_ids = np.argmax(logits, axis=-1)
            text = processor.decode(predicted_ids[0], skip_special_tokens=True)
            
            print(f"   ✅ Blank image test completed ({inference_time:.3f}s)")
            print(f"   Output: \"{text}\" (expected empty for blank image)")
        
        # Summary
        print(f"\n✅ ONNX TrOCR test completed successfully!")
        print(f"🎯 Status: Ready for use with {provider_name}")
        
        # Performance note
        if inference_time > 1.0:
            print(f"⚡ Note: Inference took {inference_time:.3f}s (CPU-based)")
        else:
            print(f"⚡ Performance: {inference_time:.3f}s per inference")
            
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)