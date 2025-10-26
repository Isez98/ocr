"""
Main application entry point - simplified and modular
"""

from src.api.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)