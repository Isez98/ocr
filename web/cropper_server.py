#!/usr/bin/env python3
"""
Manual Cropper Server
Backend server for the manual guest name cropping tool
"""

import os
import json
import base64
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_DIR = "data/manual_crops"
WEB_DIR = os.path.dirname(os.path.abspath(__file__))  # Use directory where server is located
CROPS_DATA_FILE = "data/crops_data.json"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

@app.route('/')
def serve_cropper():
    """Serve the manual cropper interface"""
    return send_from_directory(WEB_DIR, 'manual_cropper.html')

@app.route('/api/save_crop', methods=['POST'])
def save_crop():
    """Save a cropped image"""
    try:
        data = request.json
        
        # Extract data
        image_data = data['imageData']  # base64 encoded
        guest_name = data['guestName']
        file_name = data['fileName']
        dimensions = data['dimensions']
        
        # Remove data URL prefix
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Create safe filename
        safe_guest_name = "".join(c for c in guest_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_guest_name = safe_guest_name.replace(' ', '_')
        
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crop_filename = f"{Path(file_name).stem}_{safe_guest_name}_{timestamp}.png"
        crop_path = os.path.join(UPLOAD_DIR, crop_filename)
        
        # Save image file
        with open(crop_path, 'wb') as f:
            f.write(image_bytes)
        
        # Save metadata
        crop_info = {
            'id': timestamp + safe_guest_name,
            'filename': crop_filename,
            'guest_name': guest_name,
            'source_file': file_name,
            'dimensions': dimensions,
            'created_at': datetime.now().isoformat(),
            'file_path': crop_path
        }
        
        # Load existing data
        crops_data = []
        if os.path.exists(CROPS_DATA_FILE):
            with open(CROPS_DATA_FILE, 'r') as f:
                crops_data = json.load(f)
        
        # Add new crop
        crops_data.append(crop_info)
        
        # Save updated data
        with open(CROPS_DATA_FILE, 'w') as f:
            json.dump(crops_data, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': f'Crop saved as {crop_filename}',
            'crop_id': crop_info['id'],
            'file_path': crop_path
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/get_crops', methods=['GET'])
def get_crops():
    """Get all saved crops"""
    try:
        if not os.path.exists(CROPS_DATA_FILE):
            return jsonify([])
        
        with open(CROPS_DATA_FILE, 'r') as f:
            crops_data = json.load(f)
        
        return jsonify(crops_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/delete_crop', methods=['POST'])
def delete_crop():
    """Delete a saved crop"""
    try:
        data = request.json
        crop_id = data['crop_id']
        
        # Load existing data
        if not os.path.exists(CROPS_DATA_FILE):
            return jsonify({'success': False, 'error': 'No crops data found'})
        
        with open(CROPS_DATA_FILE, 'r') as f:
            crops_data = json.load(f)
        
        # Find and remove crop
        crop_to_remove = None
        for i, crop in enumerate(crops_data):
            if crop['id'] == crop_id:
                crop_to_remove = crops_data.pop(i)
                break
        
        if not crop_to_remove:
            return jsonify({'success': False, 'error': 'Crop not found'})
        
        # Delete file if it exists
        if os.path.exists(crop_to_remove['file_path']):
            os.remove(crop_to_remove['file_path'])
        
        # Save updated data
        with open(CROPS_DATA_FILE, 'w') as f:
            json.dump(crops_data, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': f'Crop {crop_id} deleted'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export_training_data', methods=['POST'])
def export_training_data():
    """Export crops as training data format"""
    try:
        # Load crops data
        if not os.path.exists(CROPS_DATA_FILE):
            return jsonify({'success': False, 'error': 'No crops data found'})
        
        with open(CROPS_DATA_FILE, 'r') as f:
            crops_data = json.load(f)
        
        if not crops_data:
            return jsonify({'success': False, 'error': 'No crops to export'})
        
        # Convert to training data format
        training_data = []
        validation_data = []
        
        for i, crop in enumerate(crops_data):
            training_item = {
                'image_path': crop['file_path'],
                'text': crop['guest_name']
            }
            
            # Split 80/20 for training/validation
            if i % 5 == 0:  # Every 5th item goes to validation
                validation_data.append(training_item)
            else:
                training_data.append(training_item)
        
        # Save training data files
        training_file = "synthetic_data/guest_names_training.json"
        validation_file = "synthetic_data/guest_names_validation.json"
        
        os.makedirs("synthetic_data", exist_ok=True)
        
        with open(training_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        with open(validation_file, 'w') as f:
            json.dump(validation_data, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': f'Exported {len(training_data)} training and {len(validation_data)} validation samples',
            'training_file': training_file,
            'validation_file': validation_file,
            'total_crops': len(crops_data)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get cropping statistics"""
    try:
        stats = {
            'total_crops': 0,
            'unique_guests': 0,
            'source_files': 0,
            'storage_used': 0
        }
        
        if os.path.exists(CROPS_DATA_FILE):
            with open(CROPS_DATA_FILE, 'r') as f:
                crops_data = json.load(f)
            
            stats['total_crops'] = len(crops_data)
            stats['unique_guests'] = len(set(crop['guest_name'] for crop in crops_data))
            stats['source_files'] = len(set(crop['source_file'] for crop in crops_data))
        
        # Calculate storage used
        if os.path.exists(UPLOAD_DIR):
            total_size = 0
            for filename in os.listdir(UPLOAD_DIR):
                filepath = os.path.join(UPLOAD_DIR, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
            stats['storage_used'] = total_size
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def create_launch_script():
    """Create a simple launch script"""
    launch_script = """#!/bin/bash
# Manual Cropper Launch Script

echo "🚀 Starting Manual Guest Name Cropper Server..."
echo "=================================="

# Activate virtual environment if it exists
if [ -f "/storage/.venv/ocr_stable_py311/bin/activate" ]; then
    source /storage/.venv/ocr_stable_py311/bin/activate
    echo "✅ Virtual environment activated"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Install Flask if not present
pip install flask flask-cors > /dev/null 2>&1

# Start server
echo "🌐 Starting server on http://localhost:5000"
echo "📁 Crops will be saved to: data/manual_crops/"
echo "💡 Press Ctrl+C to stop"
echo ""

python3 web/cropper_server.py
"""
    
    with open("web/start_cropper.sh", 'w') as f:
        f.write(launch_script)
    
    os.chmod("web/start_cropper.sh", 0o755)

if __name__ == '__main__':
    create_launch_script()
    
    print("🚀 Manual Guest Name Cropper Server")
    print("=" * 40)
    print(f"📁 Crops saved to: {os.path.abspath(UPLOAD_DIR)}")
    print(f"📊 Data file: {os.path.abspath(CROPS_DATA_FILE)}")
    print(f"🌐 Server: http://localhost:5000")
    print(f"💡 Press Ctrl+C to stop")
    print()
    
    app.run(host='localhost', port=5000, debug=True)