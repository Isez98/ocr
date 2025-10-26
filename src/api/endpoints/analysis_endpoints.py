"""
Analysis and debugging endpoints
"""

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import cv2
import io

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "OCR Analysis API",
        "version": "1.0.0"
    }

@router.post("/analyze/image-stats")
async def analyze_image_stats(file: UploadFile = File(...)):
    """Analyze basic image statistics"""
    try:
        # Read and convert image
        content = await file.read()
        pil_image = Image.open(io.BytesIO(content))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Calculate statistics
        from ..utils.image_utils import calculate_image_stats
        stats = calculate_image_stats(cv_image)
        
        return JSONResponse(content={
            "filename": file.filename,
            "image_stats": stats
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@router.get("/system/info")
async def get_system_info():
    """Get system information and loaded models"""
    try:
        import torch
        import transformers
        
        system_info = {
            "python_version": "3.x",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        return JSONResponse(content=system_info)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )