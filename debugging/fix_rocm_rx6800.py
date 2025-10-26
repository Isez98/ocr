#!/usr/bin/env python3
"""
ROCm RX 6800 Segfault Workarounds
"""

import os
import subprocess
import sys

def apply_rocm_workarounds():
    """Apply environment variables to fix RX 6800 segfaults"""
    
    print("🔧 Applying ROCm RX 6800 Workarounds...")
    
    # Known fixes for RX 6800 + ROCm 6.1 segfaults
    workarounds = {
        # Override GFX version for better compatibility
        'HSA_OVERRIDE_GFX_VERSION': '10.3.0',
        
        # Enable expandable memory segments
        'PYTORCH_HIP_ALLOC_CONF': 'expandable_segments:True',
        
        # Force synchronous execution for stability
        'HIP_LAUNCH_BLOCKING': '1',
        
        # Disable certain optimizations that cause issues
        'HIP_FORCE_DEV_KERNARG': '1',
        
        # Memory pool settings
        'HSA_ENABLE_SDMA': '0',
        
        # Disable kernel cache that can cause issues
        'HSA_DISABLE_CACHE': '1',
    }
    
    print("Setting environment variables:")
    for key, value in workarounds.items():
        os.environ[key] = value
        print(f"  {key}={value}")
    
    return workarounds

def test_with_workarounds():
    """Test basic operations with workarounds applied"""
    
    apply_rocm_workarounds()
    
    print("\n🧪 Testing with workarounds applied...")
    
    try:
        import torch
        
        print("1. Basic tensor operations...")
        x = torch.randn(10, 10, device='cuda')
        y = x + 1
        print(f"   Result shape: {y.shape}")
        
        print("2. Testing model import...")
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from PIL import Image
        
        print("3. Loading processor...")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        
        print("4. Loading model...")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        
        print("5. Moving to GPU with workarounds...")
        model = model.to('cuda')
        
        # Set required config
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        
        print("6. Testing inference with workarounds...")
        test_image = Image.new('RGB', (200, 80), 'white')
        pixel_values = processor(test_image, return_tensors="pt").pixel_values.to('cuda')
        
        # Force synchronous execution
        torch.cuda.synchronize()
        
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=16)
        
        torch.cuda.synchronize()
        
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"   Inference result: '{text}'")
        
        print("✅ Workarounds successful!")
        return True
        
    except Exception as e:
        print(f"❌ Workarounds failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_training_script_with_workarounds():
    """Create training script with all workarounds applied"""
    
    script_content = '''#!/usr/bin/env python3
"""
Training script with RX 6800 workarounds applied
"""

import os

# Apply workarounds BEFORE importing PyTorch
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HIP_LAUNCH_BLOCKING'] = '1'
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['HSA_ENABLE_SDMA'] = '0'
os.environ['HSA_DISABLE_CACHE'] = '1'

import json
import torch
import gc
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Trainer, TrainingArguments
from transformers import default_data_collator

class WorkaroundHandwritingDataset(Dataset):
    """Dataset with RX 6800 workarounds"""
    
    def __init__(self, data_file, processor, max_target_length=32):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_target_length = max_target_length
        
        # Use smaller dataset for stability
        self.data = self._filter_valid_samples()[:50]  # Very limited
        print(f"📊 Using {len(self.data)} samples with workarounds")

    def _filter_valid_samples(self):
        valid_samples = []
        for item in self.data:
            try:
                if os.path.exists(item['image_path']):
                    with Image.open(item['image_path']) as img:
                        img.convert('RGB')
                    if item['text'] and len(item['text'].strip()) > 0:
                        valid_samples.append(item)
            except Exception:
                continue
        return valid_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            with Image.open(item['image_path']) as image:
                image = image.convert('RGB')
                text = item['text']

                encoding = self.processor(image, text, 
                                         truncation=True, 
                                         padding="max_length", 
                                         max_length=self.max_target_length,
                                         return_tensors="pt")
            
            encoding = {key: val.squeeze() for key, val in encoding.items()}
            
            # Force synchronization after processing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            return encoding
            
        except Exception:
            # Minimal fallback
            dummy_image = Image.new('RGB', (100, 40), 'white')
            dummy_text = "x"
            encoding = self.processor(dummy_image, dummy_text, 
                                     truncation=True, 
                                     padding="max_length", 
                                     max_length=self.max_target_length,
                                     return_tensors="pt")
            encoding = {key: val.squeeze() for key, val in encoding.items()}
            return encoding

def train_with_workarounds():
    """Training with all RX 6800 workarounds"""
    
    data_file = "synthetic_data/training_data.json"
    if not os.path.exists(data_file):
        print("❌ Training data not found!")
        return
    
    print("🚀 Starting Training with RX 6800 Workarounds...")
    print("🔧 Workarounds applied:")
    print(f"   HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")
    print(f"   PYTORCH_HIP_ALLOC_CONF={os.environ.get('PYTORCH_HIP_ALLOC_CONF')}")
    print(f"   HIP_LAUNCH_BLOCKING={os.environ.get('HIP_LAUNCH_BLOCKING')}")
    
    # GPU setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 GPU: {torch.cuda.get_device_name()}")
        
        # Very conservative memory settings
        torch.cuda.set_per_process_memory_fraction(0.3)  # Only 30% of GPU memory
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        print("❌ No GPU found!")
        return
    
    # Load model with workarounds
    print("📥 Loading TrOCR with workarounds...")
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        
        # Set config
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        
        # Move to GPU with synchronization
        model = model.to(device)
        torch.cuda.synchronize()
        
        print(f"✅ Model loaded with workarounds")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create dataset with workarounds
    print("📊 Creating dataset with workarounds...")
    try:
        dataset = WorkaroundHandwritingDataset(data_file, processor, max_target_length=32)
        
        if len(dataset) < 5:
            print("❌ Not enough samples!")
            return
        
        # Minimal split
        train_size = max(4, int(0.8 * len(dataset)))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        print(f"🎯 Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        return
    
    # Training arguments with workarounds
    output_dir = "./trocr-workaround"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # Single sample
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=1e-5,
        logging_steps=2,
        save_steps=5,
        eval_steps=5,
        eval_strategy="steps",
        save_total_limit=1,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_pin_memory=False,
        load_best_model_at_end=False,
        dataloader_num_workers=0,
        logging_dir=f"{output_dir}/logs",
        report_to=[],
        save_safetensors=False,
        prediction_loss_only=True,
        max_steps=10,  # Very limited steps
        fp16=False,
        bf16=False,
        no_cuda=False,
        seed=42,
        disable_tqdm=False,
    )
    
    # Trainer with workarounds
    print("🏃 Setting up trainer with workarounds...")
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator,
        )
        
        # Force cleanup before training
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    except Exception as e:
        print(f"❌ Trainer setup failed: {e}")
        return
    
    # Attempt training with workarounds
    print("🚂 Starting training with workarounds (10 steps max)...")
    try:
        trainer.train()
        
        print("💾 Saving workaround model...")
        trainer.save_model()
        processor.save_pretrained(output_dir)
        
        print(f"✅ Training with workarounds completed! Saved to: {output_dir}")
        
        # Test the trained model
        print("\\n🧪 Testing workaround model...")
        test_image = Image.new('RGB', (200, 80), 'white')
        pixel_values = processor(test_image, return_tensors="pt").pixel_values.to(device)
        
        torch.cuda.synchronize()
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=32)
        torch.cuda.synchronize()
        
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"🎯 Workaround model result: '{text}'")
        
    except Exception as e:
        print(f"❌ Training with workarounds failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\\n💻 Falling back to CPU training recommended")
        return False
    
    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    return True

if __name__ == "__main__":
    try:
        success = train_with_workarounds()
        if not success:
            print("💻 GPU training unsuccessful - recommend CPU training")
    except Exception as e:
        print(f"💥 Critical error: {e}")
        print("💻 Use CPU training for reliable results")
'''
    
    with open('train_with_workarounds.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created train_with_workarounds.py")

if __name__ == "__main__":
    print("🔧 ROCm RX 6800 Troubleshooting Tool")
    print("="*50)
    
    # Test workarounds
    if test_with_workarounds():
        print("\n🎉 Workarounds successful! Creating training script...")
        create_training_script_with_workarounds()
        
        print("\n🚀 Next steps:")
        print("1. Run: python train_with_workarounds.py")
        print("2. If that fails, use CPU training: python train_reliable_cpu.py")
    else:
        print("\n💻 GPU workarounds unsuccessful - recommend CPU training")
        print("   Run: python train_reliable_cpu.py")