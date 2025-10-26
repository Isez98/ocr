#!/usr/bin/env python3
"""
Improved TrOCR training script with better parameters and data handling
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
    
    def __init__(self, data_file, processor, max_target_length=64):  # Reduced max length
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
                # Check if image exists and can be opened
                if os.path.exists(item['image_path']):
                    Image.open(item['image_path']).convert('RGB')
                    # Check if text is reasonable
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
            # Load image
            image = Image.open(item['image_path']).convert('RGB')
            text = item['text']

            # Process image and text
            encoding = self.processor(image, text, 
                                     truncation=True, 
                                     padding="max_length", 
                                     max_length=self.max_target_length,
                                     return_tensors="pt")
            
            encoding = {key: val.squeeze() for key, val in encoding.items()}
            return encoding
            
        except Exception as e:
            print(f"❌ Error processing {item['image_path']}: {e}")
            # Return a dummy sample if processing fails
            dummy_image = Image.new('RGB', (200, 80), 'white')
            dummy_text = "error"
            encoding = self.processor(dummy_image, dummy_text, 
                                     truncation=True, 
                                     padding="max_length", 
                                     max_length=self.max_target_length,
                                     return_tensors="pt")
            encoding = {key: val.squeeze() for key, val in encoding.items()}
            return encoding

def train_improved_model():
    """Train TrOCR with improved parameters"""
    
    # Check if improved data exists
    data_file = "synthetic_data/training_data.json"
    if not os.path.exists(data_file):
        print("❌ Improved training data not found!")
        print("Run: python improved_synthetic_generator.py first")
        return
    
    print("🚀 Starting Improved TrOCR Fine-tuning...")
    
    # Force CPU for stability
    device = torch.device("cpu")
    print(f"📱 Using device: {device}")
    
    # Load processor and model
    print("📥 Loading TrOCR processor and model...")
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        model = model.to(device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Set special tokens
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # Create dataset
    print("📊 Creating improved training dataset...")
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
    
    # Improved training arguments
    output_dir = "./trocr-improved"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # Smaller batch size
        per_device_eval_batch_size=1,
        num_train_epochs=5,  # More epochs
        learning_rate=1e-5,  # Lower learning rate
        logging_steps=5,
        save_steps=25,
        eval_steps=25,
        eval_strategy="steps",
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        dataloader_num_workers=0,
        use_cpu=True,
        logging_dir=f"{output_dir}/logs",
        report_to=[],  # Disable wandb logging
        save_safetensors=False,  # Use regular pytorch format
    )
    
    # Try to load metrics
    try:
        cer_metric = evaluate.load("cer")
        print("✅ CER metric loaded")
    except:
        print("⚠️  CER metric not available, training without evaluation metrics")
        cer_metric = None
    
    def compute_metrics(eval_pred):
        if cer_metric is None:
            return {}
            
        predictions, labels = eval_pred
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels
        import numpy as np
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
        
        try:
            cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
            return {"cer": cer}
        except:
            return {}
    
    # Trainer
    print("🏃 Setting up improved trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if cer_metric else None,
    )
    
    # Train
    print("🚂 Starting improved training...")
    try:
        trainer.train()
        
        # Save the model
        print("💾 Saving improved model...")
        trainer.save_model()
        processor.save_pretrained(output_dir)
        
        print(f"✅ Improved training completed! Model saved to: {output_dir}")
        
        # Quick validation
        print("\n🧪 Quick validation test...")
        test_sample_path = "synthetic_data/clean_names_0000.png"
        if os.path.exists(test_sample_path):
            from production_text_recognizer import ProductionTextRecognizer
            
            # Test new model
            recognizer = ProductionTextRecognizer(custom_model_path=output_dir)
            image = Image.open(test_sample_path)
            text, conf = recognizer.recognize_text(image)
            print(f"🎯 Test result: '{text}' (confidence: {conf:.3f})")
        
        print("\n🎉 Next steps:")
        print("1. Test the improved model: python test_model.py")
        print("2. Compare with base model: python test_model.py compare")
        print("3. If results are good, integrate: python integrate_model.py")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_improved_model()