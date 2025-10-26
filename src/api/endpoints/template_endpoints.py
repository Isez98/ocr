"""
Template management endpoints
"""

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from ..core.template_manager import TemplateManager

router = APIRouter()
template_manager = TemplateManager()

@router.get("/templates")
async def list_templates():
    """List all available templates"""
    try:
        templates = template_manager.list_templates()
        
        # Get detailed info for each template
        template_details = []
        for template_id in templates:
            info = template_manager.get_template_info(template_id)
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

@router.get("/templates/{template_id}")
async def get_template(template_id: str):
    """Get a specific template configuration"""
    try:
        config = template_manager.load_template(template_id)
        
        if not config:
            return JSONResponse(
                status_code=404,
                content={"error": "Template not found"}
            )
        
        return JSONResponse(content=config)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@router.get("/templates/{template_id}/info")
async def get_template_info(template_id: str):
    """Get information about a specific template"""
    try:
        template_info = template_manager.get_template_info(template_id)
        
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

@router.post("/templates/{template_id}/validate")
async def validate_template(template_id: str):
    """Validate a template configuration"""
    try:
        config = template_manager.load_template(template_id)
        
        if not config:
            return JSONResponse(
                status_code=404,
                content={"error": "Template not found"}
            )
        
        errors = template_manager.validate_template(config)
        
        return JSONResponse(content={
            "template_id": template_id,
            "is_valid": len(errors) == 0,
            "errors": errors
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )