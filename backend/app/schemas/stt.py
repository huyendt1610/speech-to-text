from pydantic import BaseModel


class TranscriptResponse(BaseModel):
    transcript: str
    filename: str 
    duration: float
    language: str 

# class AudioPrediction(BaseModel):
#     file: UploadFile, la: str = Form(...), model_name: str = Form(...)):