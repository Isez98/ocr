#!/usr/bin/env python3
"""
Comprehensive ONNX TrOCR Testing Script
Tests ONNX Runtime providers and TrOCR model performance
"""

import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort
from transformers import TrOCRProcessor

def create_test_image(text="Hello World", size=(384, 384)):
    """Create a test image with handwritten-style text"""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fall back to default if not available
    try:
        # Use a larger font for better recognition
        font_size = min(size) // 10
        font = ImageFont.load_default()
    except:
        font = None
    
    # Calculate text position (centered)
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        text_width, text_height = len(text) * 10, 20
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw text in black
    draw.text((x, y), text, fill='black', font=font)
    
    return img

def test_provider(model_path, provider_name, test_images):
    """Test a specific ONNX provider"""
    print(f"\n{'='*50}")
    print(f"Testing {provider_name}")
    print(f"{'='*50}")
    
    try:
        # Create session with specific provider
        print(f"1. Creating session with {provider_name}...")
        providers = [provider_name] if provider_name != 'Auto' else None
        sess = ort.InferenceSession(model_path, providers=providers)
        
        active_providers = sess.get_providers()
        print(f"   ✅ Session created successfully")
        print(f"   Active providers: {active_providers}")
        
        # Get model info
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()
        print(f"   Model inputs: {len(inputs)}")
        print(f"   Model outputs: {len(outputs)}")
        
        # Load processor
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        
        results = []
        total_time = 0
        
        for i, (image, description) in enumerate(test_images):
            print(f"\n2.{i+1} Testing with {description}...")
            
            try:
                # Process image
                pixel_values = processor(image, return_tensors='pt').pixel_values.numpy()
                decoder_input_ids = np.array([[processor.tokenizer.bos_token_id]], dtype=np.int64)
                
                # Prepare inputs
                ort_inputs = {
                    'pixel_values': pixel_values,
                    'decoder_input_ids': decoder_input_ids
                }
                
                # Run inference with timing
                start_time = time.time()
                ort_outputs = sess.run(None, ort_inputs)
                inference_time = time.time() - start_time
                total_time += inference_time
                
                # Decode result
                logits = ort_outputs[0]  # [batch, seq, vocab]
                predicted_ids = np.argmax(logits, axis=-1)
                text = processor.decode(predicted_ids[0], skip_special_tokens=True)
                
                result = {
                    'description': description,
                    'text': text,
                    'time': inference_time,
                    'output_shapes': [out.shape for out in ort_outputs]
                }
                results.append(result)
                
                print(f"     ✅ Inference successful ({inference_time:.3f}s)")
                print(f"     Recognized text: \"{text}\"")
                print(f"     Output shapes: {result['output_shapes']}")
                
            except Exception as e:
                print(f"     ❌ Inference failed: {e}")
                results.append({
                    'description': description,
                    'text': None,
                    'time': None,
                    'error': str(e)
                })
        
        # Summary
        successful_tests = len([r for r in results if 'error' not in r])
        avg_time = total_time / successful_tests if successful_tests > 0 else 0
        
        print(f"\n📊 {provider_name} Summary:")
        print(f"   Successful tests: {successful_tests}/{len(test_images)}")
        if avg_time > 0:
            print(f"   Average inference time: {avg_time:.3f}s")
            print(f"   Total time: {total_time:.3f}s")
        
        return {
            'provider': provider_name,
            'active_providers': active_providers,
            'successful': successful_tests,
            'total': len(test_images),
            'avg_time': avg_time,
            'results': results
        }
        
    except Exception as e:
        print(f"❌ Failed to create session: {e}")
        return {
            'provider': provider_name,
            'active_providers': None,
            'successful': 0,
            'total': len(test_images),
            'avg_time': None,
            'error': str(e)
        }

def main():
    print("🚀 Comprehensive ONNX TrOCR Testing")
    print("="*60)
    
    # Check ONNX Runtime
    print(f"ONNX Runtime version: {ort.__version__}")
    available_providers = ort.get_available_providers()
    print(f"Available providers: {available_providers}")
    
    # Model path
    model_path = "/home/isacc/Documents/vs-code/ocr/models/trocr.onnx"
    
    # Create test images
    test_images = [
        (create_test_image("Hello", (384, 384)), "Simple text 'Hello'"),
        (create_test_image("123", (384, 384)), "Numbers '123'"),
        (create_test_image("Test OCR", (384, 384)), "Phrase 'Test OCR'"),
        (Image.new('RGB', (384, 384), color='white'), "Blank image"),
    ]
    
    # Test providers
    providers_to_test = []
    
    # Add GPU providers if available
    if 'ROCMExecutionProvider' in available_providers:
        providers_to_test.append('ROCMExecutionProvider')
    
    if 'MIGraphXExecutionProvider' in available_providers:
        providers_to_test.append('MIGraphXExecutionProvider')
    
    # Always test CPU
    providers_to_test.append('CPUExecutionProvider')
    
    # Run tests
    all_results = []
    for provider in providers_to_test:
        result = test_provider(model_path, provider, test_images)
        all_results.append(result)
    
    # Final summary
    print(f"\n{'='*60}")
    print("🏁 FINAL SUMMARY")
    print(f"{'='*60}")
    
    for result in all_results:
        provider = result['provider']
        if 'error' in result:
            print(f"❌ {provider}: Failed to initialize - {result['error']}")
        else:
            success_rate = result['successful'] / result['total'] * 100
            avg_time = result['avg_time']
            active = result['active_providers']
            
            print(f"✅ {provider}: {result['successful']}/{result['total']} tests passed ({success_rate:.1f}%)")
            if avg_time and avg_time > 0:
                print(f"   Average time: {avg_time:.3f}s")
            print(f"   Active providers: {active}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    working_providers = [r for r in all_results if 'error' not in r and r['successful'] > 0]
    
    if working_providers:
        # Find fastest working provider
        fastest = min(working_providers, key=lambda x: x['avg_time'] if x['avg_time'] else float('inf'))
        print(f"   Best performing provider: {fastest['provider']}")
        
        # Check for GPU acceleration
        gpu_providers = [r for r in working_providers if 'CPU' not in r['provider']]
        if gpu_providers:
            print(f"   GPU acceleration available: {[r['provider'] for r in gpu_providers]}")
        else:
            print(f"   GPU acceleration: Not available, using CPU fallback")
    else:
        print(f"   No working providers found!")
    
    print(f"\n🎯 Status: ONNX TrOCR testing completed successfully!")

if __name__ == "__main__":
    main()