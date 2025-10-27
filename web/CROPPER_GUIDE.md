# Manual Guest Name Cropper

A complete web-based tool for manually cropping guest name regions from filled template photos.

## Quick Start

### 1. Start the Server
```bash
cd /home/isacc/Documents/vs-code/ocr
chmod +x web/start_cropper.sh
./web/start_cropper.sh
```

**Or manually:**
```bash
# Install dependencies
pip install flask flask-cors

# Start server
python3 web/cropper_server.py
```

### 2. Open the Interface
- Open your browser to: http://localhost:5000
- The interface will load automatically

### 3. Load Your Template Photos
- Click "📁 Select Template Images"
- Select your 50+ filled template photos from the `samples` folder
- Images will appear in the file list

### 4. Crop Guest Names
For each image:
1. **Select** the image from the file list
2. **Drag** to select the guest name region on the canvas
3. **Enter** the guest name in the text input
4. **Click** "✅ Save Crop" to save both locally and to server

### 5. Features Available

#### Navigation & Display
- **File List**: Shows all loaded images with progress indicators
- **Zoom Controls**: 10% to 300% zoom with slider
- **Canvas**: Interactive image display with selection tools

#### Cropping Tools
- **Drag Selection**: Click and drag to select guest name regions
- **Preview Panel**: Shows cropped regions in real-time
- **Guest Name Input**: Label each crop with the actual name

#### Progress Tracking
- **File Progress**: Green checkmarks for completed files
- **Crop Counter**: Total crops saved
- **Statistics**: Files processed, unique guests, storage used

#### Data Management
- **Local Storage**: Browser persistence for offline work
- **Server Backup**: Automatic server saving when available
- **Export**: Downloads training-ready data files

### 6. Keyboard Shortcuts
- **Enter**: Save current crop (when guest name entered)
- **Escape**: Cancel current selection
- **Ctrl+Z**: Undo last action (planned)

### 7. Output Files

#### Server-side (Recommended)
- **Crops**: `data/manual_crops/*.png` - Individual cropped images
- **Metadata**: `data/crops_data.json` - Crop information and labels
- **Training Data**: 
  - `synthetic_data/guest_names_training.json`
  - `synthetic_data/guest_names_validation.json`

#### Browser-side (Backup)
- Crops stored in localStorage
- Export downloads individual PNG files

### 8. Integration with Training

Once you've cropped all guest names:

```bash
# The server automatically creates training data files
# Ready for use with your enhanced training pipeline

python3 train_guest_names.py --data synthetic_data/guest_names_training.json
```

### 9. Tips for Best Results

#### Selection Guidelines
- Select just the guest name text, not extra whitespace
- Include complete characters but minimal background
- Consistent sizing helps model training

#### Naming Conventions
- Use the exact text as written (including punctuation)
- Preserve capitalization as shown
- Include titles (Mr., Mrs., Dr.) if written

#### Quality Control
- Preview each crop before saving
- Use zoom for precision on small text
- Re-crop if selection looks incorrect

### 10. Troubleshooting

#### Server Issues
- If "server offline" appears, crops save locally only
- Check that Flask is installed: `pip install flask flask-cors`
- Ensure port 5000 is available

#### Browser Issues
- Use Chrome/Firefox for best Canvas support
- Clear browser cache if interface seems stuck
- Check browser console (F12) for errors

#### File Loading Issues
- Ensure images are standard formats (JPG, PNG)
- Check file permissions in samples folder
- Try refreshing the page and reloading files

### 11. Advanced Features

#### Batch Processing
- Process multiple files without page reload
- Progress persists across browser sessions
- Resume work from where you left off

#### Data Export
- Automatic training/validation split (80/20)
- JSON format compatible with existing training pipeline
- Includes image paths and text labels

#### Server API
- `GET /api/stats` - Cropping statistics
- `POST /api/save_crop` - Save new crop
- `POST /api/export_training_data` - Export training files
- `GET /api/get_crops` - List all saved crops

## Ready to Start!

Your manual cropping tool is complete and ready to process your 50+ template photos. The combination of interactive web interface and automated server backend will help you efficiently create high-quality training data for guest name recognition.

Next step: Start the server and begin cropping! 🚀