"""
OCR processing endpoints
"""

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import cv2
import io

from ..core.ocr_processor import OCRProcessor

router = APIRouter()
ocr_processor = OCRProcessor()

@router.post("/process")
async def process_form(
    template_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Process a form image and extract field data"""
    try:
        # Read and convert image
        content = await file.read()
        pil_image = Image.open(io.BytesIO(content))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Process the form
        results = ocr_processor.process_form(cv_image, template_id)
        
        return JSONResponse(content=results)
        
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )

@router.get("/templates/{template_id}/info")
async def get_template_info(template_id: str):
    """Get information about a specific template"""
    try:
        template_info = ocr_processor.template_manager.get_template_info(template_id)
        
        if not template_info:
            return JSONResponse(
                status_code=404,
                content={"error": "Template not found"}
            )
        
        return JSONResponse(content=template_info)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@router.get("/templates")
async def list_templates():
    """List all available templates"""
    try:
        templates = ocr_processor.template_manager.list_templates()
        
        # Get detailed info for each template
        template_details = []
        for template_id in templates:
            info = ocr_processor.template_manager.get_template_info(template_id)
            if info:
                template_details.append(info)
        
        return JSONResponse(content={
            "templates": template_details,
            "total_count": len(template_details)
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )