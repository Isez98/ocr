#!/usr/bin/env python3
"""
Generate synthetic handwriting data for training
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import string
import os

class SyntheticHandwritingGenerator:
    """Generate synthetic handwriting samples"""
    
    def __init__(self):
        # Try to load handwriting-style fonts
        self.fonts = self._load_handwriting_fonts()
        
    def _load_handwriting_fonts(self):
        """Load available handwriting fonts"""
        fonts = []
        font_paths = [
            "/usr/share/fonts/truetype/liberation/",
            "/usr/share/fonts/truetype/dejavu/",
            # Add paths to handwriting fonts if available
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                for font_file in os.listdir(path):
                    if font_file.endswith('.ttf'):
                        try:
                            font = ImageFont.truetype(os.path.join(path, font_file), 40)
                            fonts.append(font)
                        except:
                            continue
        
        # Fallback to default font
        if not fonts:
            fonts = [ImageFont.load_default()]
            
        return fonts
    
    def generate_handwritten_sample(self, text: str, width: int = 200, height: int = 60):
        """Generate a handwritten-style sample"""
        
        # Create base image
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Random font
        font = random.choice(self.fonts)
        
        # Add some randomness to position
        x_offset = random.randint(5, 15)
        y_offset = random.randint(5, 15)
        
        # Draw text with slight randomness
        draw.text((x_offset, y_offset), text, fill='black', font=font)
        
        # Convert to numpy for CV2 operations
        img_array = np.array(img)
        
        # Apply handwriting-like distortions
        img_array = self._apply_handwriting_effects(img_array)
        
        return Image.fromarray(img_array)
    
    def _apply_handwriting_effects(self, img_array):
        """Apply effects to make text look more handwritten"""
        
        # Add slight rotation
        if random.random() > 0.5:
            angle = random.uniform(-3, 3)
            center = (img_array.shape[1]//2, img_array.shape[0]//2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_array = cv2.warpAffine(img_array, matrix, (img_array.shape[1], img_array.shape[0]))
        
        # Add slight perspective distortion
        if random.random() > 0.7:
            h, w = img_array.shape[:2]
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            
            # Small random distortion
            offset = random.randint(1, 3)
            pts2 = np.float32([
                [random.randint(-offset, offset), random.randint(-offset, offset)],
                [w + random.randint(-offset, offset), random.randint(-offset, offset)],
                [random.randint(-offset, offset), h + random.randint(-offset, offset)],
                [w + random.randint(-offset, offset), h + random.randint(-offset, offset)]
            ])
            
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            img_array = cv2.warpPerspective(img_array, matrix, (w, h))
        
        # Add noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
            img_array = cv2.add(img_array, noise)
        
        # Add slight blur
        if random.random() > 0.5:
            img_array = cv2.GaussianBlur(img_array, (3, 3), 0.5)
        
        return img_array
    
    def generate_training_set(self, texts: list, samples_per_text: int = 5, output_dir: str = "synthetic_data"):
        """Generate a training set with variations"""
        
        os.makedirs(output_dir, exist_ok=True)
        training_data = []
        
        for i, text in enumerate(texts):
            for j in range(samples_per_text):
                # Generate sample
                img = self.generate_handwritten_sample(text)
                
                # Save image
                filename = f"sample_{i:04d}_{j:02d}.png"
                filepath = os.path.join(output_dir, filename)
                img.save(filepath)
                
                training_data.append({
                    "image_path": filepath,
                    "text": text
                })
        
        # Save metadata
        import json
        with open(os.path.join(output_dir, "training_data.json"), "w") as f:
            json.dump(training_data, f, indent=2)
        
        return training_data

def generate_form_field_samples():
    """Generate samples specific to your form fields"""
    
    generator = SyntheticHandwritingGenerator()
    
    # Sample names (you could expand this)
    names = [
        "John Smith", "Maria Garcia", "David Johnson", "Sarah Wilson",
        "Chris Flores", "Emma Brown", "Michael Davis", "Lisa Anderson",
        "James Miller", "Jennifer Taylor", "Robert Martinez", "Ashley Jones"
    ]
    
    # Sample amounts
    amounts = [
        "$1,250.00", "$850.50", "$2,100.75", "$575.25",
        "$3,000.00", "$450.00", "$1,875.50", "$625.75"
    ]
    
    # Sample dates
    dates = [
        "12/15/2023", "01/08/2024", "03/22/2024", "06/10/2024",
        "09/05/2024", "11/18/2024", "02/14/2024", "07/30/2024"
    ]
    
    print("Generating synthetic handwriting samples...")
    
    # Generate training data
    all_texts = names + amounts + dates
    training_data = generator.generate_training_set(all_texts, samples_per_text=10)
    
    print(f"Generated {len(training_data)} training samples")
    return training_data

if __name__ == "__main__":
    generate_form_field_samples()