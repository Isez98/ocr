#!/usr/bin/env python3
"""
Ultra-conservative GPU training for AMD RX 6800 - Last attempt
"""

import json
import os
import torch
import gc
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Trainer, TrainingArguments
from transformers import default_data_collator
import psutil

def force_cleanup():
    """Force memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class UltraConservativeHandwritingDataset(Dataset):
    """Ultra-conservative dataset with minimal memory usage"""
    
    def __init__(self, data_file, processor, max_target_length=32):  # Reduced from 64
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_target_length = max_target_length
        
        # Only use subset for ultra-conservative approach
        self.data = self._filter_valid_samples()[:100]  # Reduced dataset
        print(f"📊 Using {len(self.data)} samples for ultra-conservative training")

    def _filter_valid_samples(self):
        valid_samples = []
        for item in self.data:
            try:
                if os.path.exists(item['image_path']):
                    # Pre-validate image loading
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
            # Open and immediately process to minimize memory
            with Image.open(item['image_path']) as image:
                image = image.convert('RGB')
                text = item['text']

                encoding = self.processor(image, text, 
                                         truncation=True, 
                                         padding="max_length", 
                                         max_length=self.max_target_length,
                                         return_tensors="pt")
            
            encoding = {key: val.squeeze() for key, val in encoding.items()}
            return encoding
            
        except Exception:
            # Minimal dummy data
            dummy_image = Image.new('RGB', (100, 40), 'white')  # Smaller dummy
            dummy_text = "x"
            encoding = self.processor(dummy_image, dummy_text, 
                                     truncation=True, 
                                     padding="max_length", 
                                     max_length=self.max_target_length,
                                     return_tensors="pt")
            encoding = {key: val.squeeze() for key, val in encoding.items()}
            return encoding

def train_ultra_conservative_gpu():
    """Ultra-conservative GPU training"""
    
    data_file = "synthetic_data/training_data.json"
    if not os.path.exists(data_file):
        print("❌ Training data not found!")
        return
    
    print("🚀 Starting ULTRA-CONSERVATIVE GPU Training...")
    
    # Check system resources
    print(f"💾 System RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"💾 Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    
    # GPU setup with ultra-conservative settings
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 GPU: {torch.cuda.get_device_name()}")
        
        # Ultra-conservative memory settings
        torch.cuda.set_per_process_memory_fraction(0.5)  # Only use 50% of GPU memory
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        force_cleanup()
    else:
        print("❌ No GPU found!")
        return
    
    # Load model with minimal memory
    print("📥 Loading TrOCR (minimal setup)...")
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        
        # Set required tokens
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        
        # Move to GPU carefully
        model = model.to(device)
        force_cleanup()
        
        print(f"✅ Model loaded on GPU")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Ultra-small dataset
    print("📊 Creating ultra-conservative dataset...")
    try:
        dataset = UltraConservativeHandwritingDataset(data_file, processor, max_target_length=32)
        
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
    
    # Ultra-conservative training arguments
    output_dir = "./trocr-ultra-conservative"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # Single sample per batch
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,  # Simulate larger batch
        num_train_epochs=1,  # Single epoch only
        learning_rate=5e-6,  # Very small learning rate
        logging_steps=5,
        save_steps=10,
        eval_steps=10,
        eval_strategy="steps",
        save_total_limit=1,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_pin_memory=False,
        load_best_model_at_end=False,  # Disable for simplicity
        dataloader_num_workers=0,
        logging_dir=f"{output_dir}/logs",
        report_to=[],
        save_safetensors=False,
        prediction_loss_only=True,
        max_steps=20,  # Limited steps
        fp16=False,  # Disable mixed precision
        bf16=False,
        no_cuda=False,
        seed=42,
    )
    
    # Ultra-simple trainer
    print("🏃 Setting up ultra-conservative trainer...")
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator,
        )
        
        force_cleanup()
        
    except Exception as e:
        print(f"❌ Trainer setup failed: {e}")
        return
    
    # Attempt training
    print("🚂 Starting ultra-conservative training (20 steps max)...")
    try:
        trainer.train()
        
        print("💾 Saving ultra-conservative model...")
        trainer.save_model()
        processor.save_pretrained(output_dir)
        
        print(f"✅ Ultra-conservative training completed! Saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Ultra-conservative training failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n🔄 Falling back to CPU training...")
        return False
    
    force_cleanup()
    return True

if __name__ == "__main__":
    try:
        success = train_ultra_conservative_gpu()
        if not success:
            print("💻 GPU training failed completely - use CPU training instead")
    except Exception as e:
        print(f"💥 Critical error: {e}")
        print("💻 Use CPU training for reliable results")