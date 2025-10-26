"""
Active learning pipeline for continuous model improvement
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional
import sqlite3
from PIL import Image
import hashlib

class ActiveLearningPipeline:
    """Manage active learning for OCR model improvement"""
    
    def __init__(self, db_path: str = "data/ocr_learning.db", min_samples_for_retrain: int = 50):
        self.db_path = db_path
        self.min_samples_for_retrain = min_samples_for_retrain
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for tracking corrections"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT UNIQUE,
                image_path TEXT,
                field_type TEXT,
                predicted_text TEXT,
                correct_text TEXT,
                confidence_score REAL,
                user_id TEXT,
                timestamp TEXT,
                used_for_training BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_name TEXT,
                model_path TEXT,
                training_samples_count INTEGER,
                performance_metrics TEXT,
                created_at TEXT,
                is_active BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_correction(self, image_path: str, field_type: str, predicted_text: str, 
                      correct_text: str, confidence_score: float = 0.0, user_id: str = "unknown"):
        """Log a user correction for future training"""
        
        # Create image hash for deduplication
        with open(image_path, 'rb') as f:
            image_hash = hashlib.md5(f.read()).hexdigest()
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO corrections 
                (image_hash, image_path, field_type, predicted_text, correct_text, 
                 confidence_score, user_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (image_hash, image_path, field_type, predicted_text, correct_text,
                  confidence_score, user_id, datetime.now().isoformat()))
            
            conn.commit()
            print(f"✅ Logged correction: '{predicted_text}' -> '{correct_text}'")
            
        except sqlite3.IntegrityError:
            print(f"ℹ️  Correction already exists for image {image_hash}")
        
        conn.close()
        
        # Check if we should trigger retraining
        self._check_retrain_trigger()
    
    def _check_retrain_trigger(self):
        """Check if we have enough new samples to trigger retraining"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM corrections 
            WHERE used_for_training = FALSE
        ''')
        
        unused_count = cursor.fetchone()[0]
        conn.close()
        
        if unused_count >= self.min_samples_for_retrain:
            print(f"🚀 Ready for retraining! {unused_count} new samples available.")
            return True
        else:
            print(f"📊 {unused_count}/{self.min_samples_for_retrain} samples collected for next training")
            return False
    
    def prepare_training_data(self, output_dir: str = "data/training_data") -> Dict:
        """Prepare training data from corrections"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT image_path, correct_text, field_type 
            FROM corrections 
            WHERE used_for_training = FALSE
        ''')
        
        corrections = cursor.fetchall()
        conn.close()
        
        # Copy images and prepare metadata
        training_images = []
        training_texts = []
        
        for i, (image_path, correct_text, field_type) in enumerate(corrections):
            if os.path.exists(image_path):
                # Copy image to training directory
                new_image_path = os.path.join(output_dir, f"training_{i:04d}.png")
                shutil.copy2(image_path, new_image_path)
                
                training_images.append(new_image_path)
                training_texts.append(correct_text)
        
        # Save training metadata
        training_metadata = {
            "images": training_images,
            "texts": training_texts,
            "sample_count": len(training_images),
            "created_at": datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
            json.dump(training_metadata, f, indent=2)
        
        return training_metadata
    
    def mark_samples_as_used(self):
        """Mark samples as used for training"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE corrections 
            SET used_for_training = TRUE 
            WHERE used_for_training = FALSE
        ''')
        
        conn.commit()
        conn.close()
    
    def get_performance_insights(self) -> Dict:
        """Get insights about model performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Field type accuracy
        cursor.execute('''
            SELECT field_type, COUNT(*) as error_count
            FROM corrections
            GROUP BY field_type
            ORDER BY error_count DESC
        ''')
        field_errors = cursor.fetchall()
        
        # Common error patterns
        cursor.execute('''
            SELECT predicted_text, correct_text, COUNT(*) as frequency
            FROM corrections
            GROUP BY predicted_text, correct_text
            HAVING frequency > 1
            ORDER BY frequency DESC
            LIMIT 10
        ''')
        common_errors = cursor.fetchall()
        
        # Low confidence predictions that were wrong
        cursor.execute('''
            SELECT AVG(confidence_score) as avg_confidence
            FROM corrections
            WHERE confidence_score > 0
        ''')
        avg_wrong_confidence = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "field_type_errors": field_errors,
            "common_error_patterns": common_errors,
            "average_wrong_confidence": avg_wrong_confidence,
            "insights": self._generate_insights(field_errors, common_errors)
        }
    
    def _generate_insights(self, field_errors: List, common_errors: List) -> List[str]:
        """Generate actionable insights from error patterns"""
        insights = []
        
        if field_errors:
            worst_field = field_errors[0][0]
            insights.append(f"Most errors occur in '{worst_field}' fields - consider field-specific training")
        
        if common_errors:
            most_common = common_errors[0]
            insights.append(f"Most common error: '{most_common[0]}' -> '{most_common[1]}' (occurs {most_common[2]} times)")
        
        return insights