#!/usr/bin/env python3
"""
Fine-tune TrOCR on custom handwriting data
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Trainer, TrainingArguments
from transformers import default_data_collator
import evaluate

class HandwritingDataset(Dataset):
    """Dataset for handwriting training data"""
    
    def __init__(self, image_paths, texts, processor, max_target_length=128):
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        text = self.texts[idx]

        # Process image and text
        encoding = self.processor(image, text, 
                                 truncation=True, 
                                 padding="max_length", 
                                 max_length=self.max_target_length,
                                 return_tensors="pt")
        
        encoding = {key: val.squeeze() for key, val in encoding.items()}
        return encoding

def create_training_data_from_corrections(corrections_file: str):
    """
    Create training data from user corrections
    
    corrections_file format:
    [
        {
            "image_path": "path/to/roi_patch.png",
            "predicted_text": "what_model_predicted", 
            "correct_text": "what_it_should_be",
            "field_type": "text|date|currency|digits"
        }
    ]
    """
    with open(corrections_file, 'r') as f:
        corrections = json.load(f)
    
    image_paths = []
    texts = []
    
    for correction in corrections:
        if os.path.exists(correction['image_path']):
            image_paths.append(correction['image_path'])
            texts.append(correction['correct_text'])
    
    return image_paths, texts

def fine_tune_trocr(train_image_paths, train_texts, val_image_paths=None, val_texts=None, 
                   output_dir="./trocr-finetuned", epochs=5):
    """Fine-tune TrOCR on custom data"""
    
    # Load processor and model
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    
    # Set special tokens
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # Create datasets
    train_dataset = HandwritingDataset(train_image_paths, train_texts, processor)
    val_dataset = None
    if val_image_paths and val_texts:
        val_dataset = HandwritingDataset(val_image_paths, val_texts, processor)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps" if val_dataset else "no",
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_pin_memory=False,
        load_best_model_at_end=True if val_dataset else False,
    )
    
    # Metrics
    cer_metric = evaluate.load("cer")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
        
        cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        return {"cer": cer}
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model()
    processor.save_pretrained(output_dir)
    
    return output_dir

def collect_training_data_from_usage():
    """
    Set up automatic data collection for future training
    This would be integrated into your main OCR pipeline
    """
    corrections_data = []
    
    # This would be called when user provides corrections
    def log_correction(image_path, predicted_text, correct_text, field_type):
        corrections_data.append({
            "image_path": image_path,
            "predicted_text": predicted_text,
            "correct_text": correct_text,
            "field_type": field_type,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save to file
        with open("corrections_log.json", "w") as f:
            json.dump(corrections_data, f, indent=2)
    
    return log_correction

if __name__ == "__main__":
    # Example usage
    print("TrOCR Fine-tuning Setup")
    print("1. Collect training data from user corrections")
    print("2. Run fine-tuning with: python train_trocr.py")
    
    # Example training (would need actual data)
    # corrections_file = "user_corrections.json"
    # if os.path.exists(corrections_file):
    #     train_images, train_texts = create_training_data_from_corrections(corrections_file)
    #     if len(train_images) > 10:  # Need minimum amount of data
    #         fine_tune_trocr(train_images, train_texts)
    #         print("Fine-tuning completed!")