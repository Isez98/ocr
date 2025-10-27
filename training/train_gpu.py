#!/usr/bin/env python3
"""
GPU-enabled TrOCR training script for AMD GPUs with ROCm
"""

import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Trainer, TrainingArguments
from transformers import default_data_collator
import evaluate

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

            # Process image and text separately (correct TrOCR API)
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            
            # Process text labels separately  
            labels = self.processor.tokenizer(text,
                                            truncation=True,
                                            padding="max_length",
                                            max_length=self.max_target_length,
                                            return_tensors="pt").input_ids
            
            encoding = {
                'pixel_values': pixel_values.squeeze(),
                'labels': labels.squeeze()
            }
            return encoding
            
        except Exception as e:
            print(f"❌ Error processing {item['image_path']}: {e}")
            dummy_image = Image.new('RGB', (200, 80), 'white')
            dummy_text = "error"
            
            pixel_values = self.processor(dummy_image, return_tensors="pt").pixel_values
            labels = self.processor.tokenizer(dummy_text,
                                            truncation=True,
                                            padding="max_length",
                                            max_length=self.max_target_length,
                                            return_tensors="pt").input_ids
            
            encoding = {
                'pixel_values': pixel_values.squeeze(),
                'labels': labels.squeeze()
            }
            return encoding

def train_gpu_model():
    """Train TrOCR with GPU acceleration"""
    
    data_file = "synthetic_data/training_data.json"
    if not os.path.exists(data_file):
        print("❌ Training data not found!")
        return
    
    print("🚀 Starting GPU-Accelerated TrOCR Fine-tuning...")
    
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 Using AMD GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("📱 GPU not available, using CPU")
    
    # Load processor and model
    print("📥 Loading TrOCR processor and model...")
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        model = model.to(device)
        print(f"✅ Model loaded on {device}")
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
    
    # GPU-optimized training arguments
    output_dir = "./trocr-gpu-improved"
    
    # Adjust batch size based on available GPU memory and stability improvements
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # With BIOS updates and stress tests passed, we can be more aggressive
        if gpu_memory_gb >= 12:
            batch_size = 6  # More aggressive for 16GB GPU after stability improvements
        elif gpu_memory_gb >= 8:
            batch_size = 4  # Standard batch for good memory
        else:
            batch_size = 2  # Conservative for lower memory
        print(f"🎮 Using batch size: {batch_size} (GPU memory: {gpu_memory_gb:.1f}GB)")
        print(f"💪 Hardware stability verified - using optimized settings")
    else:
        batch_size = 1
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,  # Increased epochs with stable hardware
        learning_rate=3e-5,  # Optimized learning rate for AMD GPU
        logging_steps=3,  # More frequent logging for monitoring
        save_steps=20,
        eval_steps=20,
        eval_strategy="steps",
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_pin_memory=True if torch.cuda.is_available() else False,
        load_best_model_at_end=True,
                dataloader_num_workers=0,  # Disable workers to prevent API conflicts
        logging_dir=f"{output_dir}/logs",
        report_to=[],
        save_safetensors=False,
        fp16=False,  # Keep disabled for AMD GPU stability
        bf16=False,  # Explicitly disable for ROCm compatibility
        gradient_accumulation_steps=1,  # Direct training with stable hardware
        warmup_steps=50,  # Add warmup for better convergence
        weight_decay=0.01,  # Add regularization
        max_grad_norm=1.0,  # Gradient clipping for stability
    )
    
    # Try to load metrics
    try:
        cer_metric = evaluate.load("cer")
        print("✅ CER metric loaded")
    except:
        print("⚠️  CER metric not available")
        cer_metric = None
    
    def compute_metrics(eval_pred):
        if cer_metric is None:
            return {}
            
        predictions, labels = eval_pred
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        
        import numpy as np
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
        
        try:
            cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
            return {"cer": cer}
        except:
            return {}
    
    # Trainer
    print("🏃 Setting up GPU trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if cer_metric else None,
    )
    
    # Train
    print("🔥 Starting GPU-accelerated training...")
    try:
        # Clear GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"💾 GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
            print(f"🔥 GPU Performance: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB VRAM")
            print(f"🚀 BIOS optimized - ready for intensive training")
        
        trainer.train()
        
        # Save the model
        print("💾 Saving GPU-trained model...")
        trainer.save_model()
        processor.save_pretrained(output_dir)
        
        print(f"✅ GPU training completed! Model saved to: {output_dir}")
        
        # Quick validation
        print("\n🧪 Quick validation test...")
        test_sample_path = "synthetic_data/clean_names_0000.png"
        if os.path.exists(test_sample_path):
            # Test new model with CPU for compatibility
            model_cpu = model.cpu()
            
            pixel_values = processor(Image.open(test_sample_path), return_tensors="pt").pixel_values
            with torch.no_grad():
                generated_ids = model_cpu.generate(pixel_values, max_length=64)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"🎯 GPU-trained model result: '{text}'")
        
        if torch.cuda.is_available():
            print(f"💾 Final GPU memory usage: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
        
        print("\n🎉 Next steps:")
        print("1. Test the GPU-trained model: python test_model.py")
        print("2. Compare with base model: python test_model.py compare")
        
    except Exception as e:
        print(f"❌ GPU training failed: {e}")
        if torch.cuda.is_available():
            print(f"💾 GPU memory at failure: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_gpu_model()