"""
Machine learning and training endpoints
"""

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
import os

from ..ml.active_learning import ActiveLearningPipeline

router = APIRouter()
learning_pipeline = ActiveLearningPipeline()

@router.post("/submit-correction")
async def submit_correction(
    field_name: str = Form(...),
    predicted_text: str = Form(...),
    correct_text: str = Form(...),
    confidence: float = Form(0.0),
    user_id: str = Form("api_user"),
    file: UploadFile = File(...)
):
    """Submit a correction for model training"""
    try:
        # Save the image temporarily
        temp_path = f"temp_corrections/{file.filename}"
        os.makedirs("temp_corrections", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Log the correction
        learning_pipeline.log_correction(
            image_path=temp_path,
            field_type=field_name,
            predicted_text=predicted_text,
            correct_text=correct_text,
            confidence_score=confidence,
            user_id=user_id
        )
        
        return JSONResponse({
            "status": "success",
            "message": "Correction logged successfully",
            "field_name": field_name,
            "correction": f"'{predicted_text}' -> '{correct_text}'"
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@router.get("/training-status")
async def get_training_status():
    """Get current training status and insights"""
    try:
        insights = learning_pipeline.get_performance_insights()
        
        # Check how many samples we have for retraining
        import sqlite3
        conn = sqlite3.connect(learning_pipeline.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM corrections WHERE used_for_training = FALSE")
        unused_samples = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM corrections")
        total_samples = cursor.fetchone()[0]
        
        conn.close()
        
        return JSONResponse({
            "status": "success",
            "training_ready": unused_samples >= learning_pipeline.min_samples_for_retrain,
            "unused_samples": unused_samples,
            "total_samples": total_samples,
            "min_samples_needed": learning_pipeline.min_samples_for_retrain,
            "insights": insights
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@router.post("/trigger-training")
async def trigger_training():
    """Trigger model retraining with collected corrections"""
    try:
        # Check if we have enough samples
        import sqlite3
        conn = sqlite3.connect(learning_pipeline.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM corrections WHERE used_for_training = FALSE")
        unused_samples = cursor.fetchone()[0]
        conn.close()
        
        if unused_samples < learning_pipeline.min_samples_for_retrain:
            return JSONResponse({
                "status": "error",
                "message": f"Not enough samples. Need {learning_pipeline.min_samples_for_retrain}, have {unused_samples}"
            })
        
        # Prepare training data
        training_metadata = learning_pipeline.prepare_training_data()
        
        # Mark samples as used
        learning_pipeline.mark_samples_as_used()
        
        return JSONResponse({
            "status": "success",
            "message": "Training data prepared. Run the training script to fine-tune the model.",
            "training_samples": training_metadata["sample_count"],
            "next_steps": [
                "1. Install training dependencies: pip install -r requirements_training.txt",
                "2. Run training: python -m src.ml.train_model",
                "3. Update model path in configuration to use fine-tuned model"
            ]
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )