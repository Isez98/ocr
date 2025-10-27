#!/bin/bash
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
