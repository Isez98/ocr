"""
FastAPI application with modular endpoints
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .endpoints import ocr_endpoints, learning_endpoints, template_endpoints, analysis_endpoints

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="OCR Service API",
        description="Modular OCR processing service with machine learning capabilities",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include endpoint routers
    app.include_router(ocr_endpoints.router, prefix="/api/v1", tags=["OCR"])
    app.include_router(learning_endpoints.router, prefix="/api/v1", tags=["Learning"])
    app.include_router(template_endpoints.router, prefix="/api/v1", tags=["Templates"])
    app.include_router(analysis_endpoints.router, prefix="/api/v1", tags=["Analysis"])
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "OCR API"}
    
    return app

app = create_app()