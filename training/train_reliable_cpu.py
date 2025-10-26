#!/usr/bin/env python3
"""
Reliable CPU training with improved synthetic data - fallback option
"""

import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Trainer, TrainingArguments
from transformers import default_data_collator

class ImprovedHandwritingDataset(Dataset):
    """Dataset using improved synthetic data"""
    
    def __init__(self, data_file, processor, max_target_length=64):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_target_length = max_target_length
        
        self.data = self._filter_valid_samples()
        print(f"📊 Loaded {len(self.data)} valid training samples")

    def _filter_valid_samples(self):
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
            # Return dummy data if processing fails
            dummy_image = Image.new('RGB', (200, 80), 'white')
            dummy_text = "error"
            encoding = self.processor(dummy_image, dummy_text, 
                                     truncation=True, 
                                     padding="max_length", 
                                     max_length=self.max_target_length,
                                     return_tensors="pt")
            encoding = {key: val.squeeze() for key, val in encoding.items()}
            return encoding

def train_reliable_cpu():
    """Reliable CPU training with improved data"""
    
    data_file = "synthetic_data/training_data.json"
    if not os.path.exists(data_file):
        print("❌ Improved training data not found!")
        return
    
    print("🚀 Starting Reliable CPU Training with Improved Data...")
    
    # Force CPU for maximum reliability
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
    
    # Optimized CPU training arguments
    output_dir = "./trocr-reliable-cpu"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,  # Slightly larger for CPU
        per_device_eval_batch_size=2,
        num_train_epochs=3,  # More epochs for better results
        learning_rate=1e-5,  # Conservative learning rate
        logging_steps=10,
        save_steps=30,
        eval_steps=30,
        eval_strategy="steps",
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        dataloader_num_workers=0,
        use_cpu=True,
        logging_dir=f"{output_dir}/logs",
        report_to=[],
        save_safetensors=False,
        prediction_loss_only=True,  # Simplify for reliability
    )
    
    # Trainer
    print("🏃 Setting up reliable CPU trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )
    
    # Train
    print("🚂 Starting reliable CPU training...")
    try:
        trainer.train()
        
        # Save the model
        print("💾 Saving reliable CPU model...")
        trainer.save_model()
        processor.save_pretrained(output_dir)
        
        print(f"✅ Reliable CPU training completed! Model saved to: {output_dir}")
        
        # Quick validation
        print("\n🧪 Quick validation test...")
        test_sample_path = "synthetic_data/clean_names_0000.png"
        if os.path.exists(test_sample_path):
            pixel_values = processor(Image.open(test_sample_path), return_tensors="pt").pixel_values
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_length=64)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"🎯 CPU-trained model result: '{text}'")
        
        print("\n🎉 Next steps:")
        print("1. Test the reliable model: python test_model.py")
        print("2. Update production config to use reliable CPU model")
        print("3. Deploy with confidence!")
        
    except Exception as e:
        print(f"❌ CPU training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_reliable_cpu()