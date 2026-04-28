from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from app.schemas.stt import TranscriptResponse
from app.services.stt_service import run_inference

router = APIRouter() 

@router.post("/predict", response_model=TranscriptResponse)
async def predict(file: UploadFile = File(...), la: str = Form(...), model_name: str = Form(...)):
    try:
        audio_bytes = await file.read()
        pred_transcript, language, duration = run_inference(model_name, la, audio_bytes)
        
        return {
            "transcript": pred_transcript,
            "filename": file.filename,
            "duration": duration,
            "language": language
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Model inference error")
