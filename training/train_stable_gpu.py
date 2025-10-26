#!/usr/bin/env python3
"""
Stable GPU training script with segfault protection for AMD GPUs
"""

import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Trainer, TrainingArguments
from transformers import default_data_collator
import evaluate
import gc

class ImprovedHandwritingDataset(Dataset):
    """Improved dataset with better error handling"""
    
    def __init__(self, data_file, processor, max_target_length=64):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_target_length = max_target_length
        
        # Filter out any problematic samples
        self.data = self._filter_valid_samples()
        print(f"📊 Loaded {len(self.data)} valid training samples")

    def _filter_valid_samples(self):
        """Filter out samples with issues"""
        valid_samples = []
        for item in self.data:
            try:
                if os.path.exists(item['image_path']):
                    Image.open(item['image_path']).convert('RGB')
                    if item['text'] and len(item['text'].strip()) > 0:
                        valid_samples.append(item)
            except Exception as e:
                print(f"⚠️  Skipping {item['image_path']}: {e}")
        return valid_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            image = Image.open(item['image_path']).convert('RGB')
            text = item['text']

            encoding = self.processor(image, text, 
                                     truncation=True, 
                                     padding="max_length", 
                                     max_length=self.max_target_length,
                                     return_tensors="pt")
            
            encoding = {key: val.squeeze() for key, val in encoding.items()}
            return encoding
            
        except Exception as e:
            print(f"❌ Error processing {item['image_path']}: {e}")
            dummy_image = Image.new('RGB', (200, 80), 'white')
            dummy_text = "error"
            encoding = self.processor(dummy_image, dummy_text, 
                                     truncation=True, 
                                     padding="max_length", 
                                     max_length=self.max_target_length,
                                     return_tensors="pt")
            encoding = {key: val.squeeze() for key, val in encoding.items()}
            return encoding

def train_stable_gpu():
    """Train with stable GPU settings to avoid segfaults"""
    
    data_file = "synthetic_data/training_data.json"
    if not os.path.exists(data_file):
        print("❌ Training data not found!")
        return
    
    print("🚀 Starting Stable GPU TrOCR Training...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 AMD GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"💾 Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Clear any existing GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 Cleared GPU cache")
        
        # Set conservative memory settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        print("⚙️  Set conservative memory allocation")
        
    else:
        device = torch.device("cpu")
        print("📱 No GPU available, using CPU")
    
    # Load processor and model with error handling
    print("📥 Loading TrOCR processor and model...")
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        
        # Move to GPU carefully
        if device.type == "cuda":
            print("🔄 Moving model to GPU...")
            model = model.to(device)
            torch.cuda.synchronize()  # Wait for GPU operations to complete
            print(f"✅ Model successfully loaded on GPU")
            print(f"💾 GPU memory after model load: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        else:
            print("✅ Model loaded on CPU")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Set special tokens
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # Create dataset
    print("📊 Creating training dataset...")
    full_dataset = ImprovedHandwritingDataset(data_file, processor)
    
    if len(full_dataset) < 10:
        print("❌ Not enough valid training samples!")
        return
    
    # Split data (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"🎯 Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    # Conservative GPU settings to avoid segfaults
    output_dir = "./trocr-stable-gpu"
    
    if device.type == "cuda":
        # Very conservative batch size to avoid memory issues
        batch_size = 2  # Start small
        dataloader_workers = 0  # Avoid multiprocessing issues
        pin_memory = False  # Disable for stability
        print(f"🎮 Conservative GPU settings - Batch size: {batch_size}")
    else:
        batch_size = 1
        dataloader_workers = 0
        pin_memory = False
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=2,  # Reduce epochs to avoid long GPU sessions
        learning_rate=1e-5,  # Conservative learning rate
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_pin_memory=pin_memory,
        load_best_model_at_end=True,
        dataloader_num_workers=dataloader_workers,
        logging_dir=f"{output_dir}/logs",
        report_to=[],
        save_safetensors=False,
        fp16=False,  # Disable half precision - can cause issues on AMD
        gradient_accumulation_steps=1,  # No accumulation for simplicity
        # Additional stability settings
        prediction_loss_only=True,  # Simplify metrics
        no_cuda=False if device.type == "cuda" else True,
    )
    
    # Skip metrics for stability
    print("⚠️  Skipping CER metrics for stability")
    
    # Trainer with minimal configuration
    print("🏃 Setting up stable trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        # No metrics to avoid complications
    )
    
    # Train with careful error handling
    print("🔥 Starting stable GPU training...")
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            print(f"💾 GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        # Start training
        trainer.train()
        
        if device.type == "cuda":
            torch.cuda.synchronize()  # Wait for all operations
        
        # Save the model
        print("💾 Saving stable GPU model...")
        trainer.save_model()
        processor.save_pretrained(output_dir)
        
        print(f"✅ Stable GPU training completed! Model saved to: {output_dir}")
        
        # Quick validation test
        print("\n🧪 Quick validation test...")
        test_sample_path = "synthetic_data/clean_names_0000.png"
        if os.path.exists(test_sample_path):
            try:
                # Move model back to CPU for testing to avoid GPU issues
                model_cpu = model.cpu()
                pixel_values = processor(Image.open(test_sample_path), return_tensors="pt").pixel_values
                
                with torch.no_grad():
                    generated_ids = model_cpu.generate(pixel_values, max_length=64)
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print(f"🎯 GPU-trained model result: '{text}'")
                
            except Exception as e:
                print(f"⚠️  Validation test failed: {e}")
        
        if device.type == "cuda":
            print(f"💾 Final GPU memory usage: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            torch.cuda.empty_cache()
        
        print("\n🎉 Next steps:")
        print("1. Test: python test_model.py")
        print("2. Compare: python test_model.py compare")
        
    except Exception as e:
        print(f"❌ Stable GPU training failed: {e}")
        if device.type == "cuda":
            print(f"💾 GPU memory at failure: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            torch.cuda.empty_cache()
        
        print("\n💡 Fallback recommendation:")
        print("   Try CPU training: python train_improved.py")
        
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_stable_gpu()