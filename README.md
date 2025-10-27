# OCR Service - Modular Architecture

A modern, modular OCR processing service with machine learning capabilities.

## 📁 Project Structure

```
ocr/
├── main.py                     # Application entry point
├── requirements*.txt           # Dependencies
├── Dockerfile                 # Container configuration
│
├── src/                       # Source code
│   ├── api/                   # API layer
│   │   ├── main.py           # FastAPI app factory
│   │   └── endpoints/        # API endpoints
│   │       ├── ocr_endpoints.py
│   │       ├── learning_endpoints.py
│   │       ├── template_endpoints.py
│   │       └── analysis_endpoints.py
│   │
│   ├── core/                  # Core business logic
│   │   ├── ocr_processor.py   # Main OCR pipeline
│   │   ├── image_processor.py # Image processing
│   │   ├── text_recognizer.py # TrOCR text recognition
│   │   └── template_manager.py# Template management
│   │
│   ├── ml/                    # Machine learning
│   │   ├── active_learning.py # Learning pipeline
│   │   ├── train_trocr.py    # Model training
│   │   └── synthetic_data_generator.py
│   │
│   └── utils/                 # Utilities
│       ├── image_utils.py     # Image utility functions
│       └── alignment_utils.py # Alignment utilities
│
├── data/                      # Data storage
│   ├── templates/            # Template configurations
│   ├── training_data/        # ML training data
│   └── temp_corrections/     # Temporary correction files
│
├── tools/                     # Development tools
│   ├── roi_analysis.py       # ROI analysis tools
│   ├── roi_selector.py       # ROI selection tools
│   ├── snap_to_ink.py       # Template refinement
│   └── optimize_rois.py     # ROI optimization
│
├── training/                  # Model training scripts
│   ├── train_gpu.py          # GPU training (ROCm/CUDA)
│   └── README.md             # Training documentation
│
├── debugging/                 # Debugging and troubleshooting
│   ├── debug_rocm.py         # ROCm diagnostics
│   ├── fix_rocm_rx6800.py    # AMD GPU workarounds
│   └── README.md             # Debug documentation
│
├── models/                    # Model management
│   ├── test_model.py         # Model testing and comparison
│   ├── production_text_recognizer.py  # Production OCR
│   ├── integrate_model.py    # Model deployment
│   └── README.md             # Model documentation
│
├── tests/                     # Tests and debugging
│   ├── test_*.py            # Unit tests
│   ├── debug_*.py           # Debug scripts
│   └── demo_*.py            # Demo scripts
│
└── web/                       # Web interfaces
    ├── roi_editor.html       # ROI editor
    └── snap_to_ink_ui.html   # Template refinement UI
```

## 🚀 Getting Started

### Quick Start

1. **Environment Setup**:
   ```bash
   # Use the pre-configured environment (recommended for AMD GPU users)
   source /storage/venv_profile.sh && activate_ocr
   
   # Or create new environment
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start the service**:
   ```bash
   python main.py
   ```

3. **Docker deployment**:
   ```bash
   docker build -t ocr-service .
   docker run -p 8001:8001 ocr-service
   ```

### API Usage
```bash
# Process a form
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "template_id=form_v1" \
  -F "file=@sample.jpg"

# Submit correction for training
curl -X POST "http://localhost:8001/api/v1/submit-correction" \
  -F "field_name=guest_name" \
  -F "predicted_text=Chris" \
  -F "correct_text=Chris Flores" \
  -F "file=@correction.jpg"

# Check training status
curl -X GET "http://localhost:8001/api/v1/training-status"
```

## 🏗️ Architecture Benefits

### Modular Design
- **Separation of Concerns**: Each module has a single responsibility
- **Easy Testing**: Isolated components can be tested independently
- **Maintainable**: Changes to one module don't affect others

### Scalable Structure
- **API Layer**: Clean REST endpoints with proper routing
- **Core Logic**: Business logic separated from presentation
- **ML Pipeline**: Machine learning components isolated for easy updates

### Development Workflow
- **Tools Separated**: Development tools don't clutter main code
- **Data Organized**: Templates, training data, and temporary files properly organized
- **Tests Isolated**: All testing code in dedicated directory

## 📊 API Endpoints

### OCR Processing
- `POST /api/v1/process` - Process form images
- `GET /api/v1/templates` - List available templates
- `GET /api/v1/templates/{id}/info` - Get template information

### Machine Learning
- `POST /api/v1/submit-correction` - Submit training corrections
- `GET /api/v1/training-status` - Check training progress
- `POST /api/v1/trigger-training` - Start model retraining

### Analysis & Debugging
- `POST /api/v1/analyze/image-stats` - Analyze image statistics
- `GET /api/v1/system/info` - Get system information
- `GET /health` - Health check

## 🔧 Development

### Adding New Features
1. **Core Logic**: Add to `src/core/` if it's business logic
2. **API Endpoints**: Add to `src/api/endpoints/` for new endpoints
3. **ML Features**: Add to `src/ml/` for machine learning
4. **Utilities**: Add to `src/utils/` for shared utilities

### Testing
```bash
# Run specific tests
python -m pytest tests/test_ocr_processor.py

# Run all tests
python -m pytest tests/
```

### Tools
```bash
# Analyze ROI coverage
python tools/roi_analysis.py

# Optimize ROI positions
python tools/optimize_rois.py

# Interactive ROI editor
open web/roi_editor.html
```

## 📈 Machine Learning

### Training Workflow
1. Use the service normally
2. Submit corrections when OCR is wrong
3. System automatically tracks errors and patterns
4. When enough data is collected (50+ samples), trigger retraining
5. Updated model improves accuracy for your specific handwriting

### Model Management
- **Base Model**: Microsoft TrOCR (handwritten text)
- **Fine-tuning**: Custom training on your corrections
- **Version Control**: Track model versions and performance
- **Rollback**: Easy rollback to previous model versions

## 🔍 Monitoring

### Performance Insights
- Field-type error analysis
- Common error patterns
- Confidence score tracking
- Training data quality metrics

### Health Monitoring
- API health checks
- Model loading status
- System resource usage
- Processing performance metrics