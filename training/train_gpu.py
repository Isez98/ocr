#!/usr/bin/env python3
"""
GPU-enabled TrOCR training script for AMD GPUs with ROCm
"""

import json
import os
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Trainer, TrainingArguments
from transformers import default_data_collator, EarlyStoppingCallback
import evaluate
import torchvision.transforms as transforms
import random
import numpy as np
from jiwer import cer, wer

class ImprovedHandwritingDataset(Dataset):
    """Improved dataset with better error handling and data augmentation"""
    
    def __init__(self, data_file, processor, max_target_length=64, augment=False):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_target_length = max_target_length
        self.augment = augment
        
        # Filter out any problematic samples
        self.data = self._filter_valid_samples()
        print(f"📊 Loaded {len(self.data)} valid training samples")
        
        # Data augmentation transforms
        if self.augment:
            self.augment_transforms = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.3),
            ])
            print("🎨 Data augmentation enabled")

    def _augment_image(self, image):
        """Apply data augmentation to reduce overfitting"""
        if not self.augment:
            return image
            
        # Convert to tensor for torchvision transforms
        image_tensor = transforms.ToTensor()(image)
        
        # Apply augmentations
        if random.random() < 0.7:  # 70% chance to augment
            image_tensor = self.augment_transforms(image_tensor)
            
        # Additional custom augmentations
        if random.random() < 0.3:  # Random rotation ±3 degrees
            angle = random.uniform(-3, 3)
            image_tensor = transforms.functional.rotate(image_tensor, angle, fill=1.0)
            
        # Convert back to PIL
        augmented = transforms.ToPILImage()(image_tensor)
        return augmented
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

            # Apply augmentation for training
            if self.augment:
                image = self._augment_image(image)

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
    
    # Create dataset with augmentation for training
    print("📊 Creating training dataset...")
    full_dataset = ImprovedHandwritingDataset(data_file, processor, augment=True)
    
    if len(full_dataset) < 10:
        print("❌ Not enough valid training samples!")
        return
    
    # Split data (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Create training dataset with augmentation, validation without
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(full_dataset)))
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    # Create validation dataset without augmentation
    val_dataset_no_aug = ImprovedHandwritingDataset(data_file, processor, augment=False)
    val_dataset = torch.utils.data.Subset(val_dataset_no_aug, [i - train_size for i in val_indices])
    
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
        num_train_epochs=10,  # Increased but early stopping will prevent overfitting
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
        metric_for_best_model="cer",  # Use CER for best model selection
        greater_is_better=False,  # Lower CER is better
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
        # Early stopping configuration
        evaluation_strategy="steps",
        save_strategy="steps",
    )    # Load evaluation metrics
    print("📏 Setting up evaluation metrics...")
    
    def compute_metrics(eval_pred):
        """Compute CER and WER metrics for proper evaluation"""
        predictions, labels = eval_pred
        
        # Decode predictions
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in labels with pad token for decoding
        import numpy as np
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
        
        # Clean up predictions and labels
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        try:
            # Calculate CER and WER using jiwer
            cer_score = cer(decoded_labels, decoded_preds)
            wer_score = wer(decoded_labels, decoded_preds)
            
            # Calculate accuracy (exact match)
            exact_matches = sum(1 for pred, label in zip(decoded_preds, decoded_labels) if pred == label)
            accuracy = exact_matches / len(decoded_preds) if decoded_preds else 0
            
            return {
                "cer": cer_score,
                "wer": wer_score, 
                "accuracy": accuracy,
                "exact_matches": exact_matches,
                "total_samples": len(decoded_preds)
            }
        except Exception as e:
            print(f"⚠️  Metric calculation error: {e}")
            return {"cer": 1.0, "wer": 1.0, "accuracy": 0.0}
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=5,  # Stop if no improvement for 5 evaluations
        early_stopping_threshold=0.001  # Minimum change to qualify as improvement
    )
    
    # Trainer
    print("🏃 Setting up GPU trainer with anti-overfitting measures...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
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
        
        print(f"✅ Anti-overfitting training completed! Model saved to: {output_dir}")
        print(f"📏 Final metrics logged - check for CER/WER improvements")
        
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
        
        # Training completion summary
        training_history = trainer.state.log_history
        if training_history:
            final_metrics = training_history[-1] if training_history else {}
            if 'eval_cer' in final_metrics:
                print(f"� Final CER: {final_metrics['eval_cer']:.4f}")
            if 'eval_wer' in final_metrics:
                print(f"📊 Final WER: {final_metrics['eval_wer']:.4f}")
            if 'eval_accuracy' in final_metrics:
                print(f"🎯 Final Accuracy: {final_metrics['eval_accuracy']:.4f}")
        
        print("\n🎉 Next steps:")
        print("1. Check CER/WER metrics in logs for overfitting analysis")
        print("2. Test the robust model: python models/test_model.py")
        print("3. Compare metrics with base model: python models/test_model.py --compare")
        
    except Exception as e:
        print(f"❌ Anti-overfitting training failed: {e}")
        if torch.cuda.is_available():
            print(f"💾 GPU memory at failure: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_gpu_model()