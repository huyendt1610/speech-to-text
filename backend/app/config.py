from pydantic_settings import BaseSettings, SettingsConfigDict 
from typing import List 
import json 
import os 
import torch

_debug_log_path = os.environ.get("DEBUG_LOG_PATH", "/app/logs/debug.log")
try:
    os.makedirs(os.path.dirname(_debug_log_path), exist_ok=True)
    with open(_debug_log_path, "a") as _f: 
        _f.write(
            json.dumps({
                "location":"config.py:9",
                "message": "Config Module Loading",
                "data": {"version": "v3_model_config"}
            })
            + "\n"
        )
except Exception: 
    pass 

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        cache_strings=True, 
        env_ignore_empty=True, 
        extra="ignore"
    )

    DATABASE_URL: str = "sqlite:///./app/db/stt.db"

    # JWT 
    JWT_SECRET_KEY: str = "your-secret-key-here"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_DAYS: int = 7
    REMEMBER_ME_EXPIRE_DAYS: int = 30

    # Email
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587 
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM: str = "noreply@stt.com"

    #CORS
    CORS_ORIGINS: str = "http://localhost:3000"

    # File upload 
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 5242880  # 5MB

    # STT 
    SAMPLING_RATE: int = 16000
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    WAV2VEC2_FINNISH: str = "jonatasgrosman/wav2vec2-large-xlsr-53-finnish"
    WAV2VEC2_BASE: str = "facebook/wav2vec2-base-960h"

    DEEPSPEECH2_MODEL_PATH: str="../ml/saved_models/best_weights.pt"
    DEEPSPEECH2_CONFIG_PATH: str="../ml/saved_models/config.json"

    WHISPER_MODEL_NAME: str = "tiny"

    @property 
    def cors_origins_list(self) -> List[str]: 
        """Parse CORS_ORIGINS into a list. Supports JSON array or comma-seperated string"""
        if not self.CORS_ORIGINS or self.CORS_ORIGINS.strip() == "":
            return ["http://localhost:3000"]

        # try parsing as JSON 
        try:
            parsed = json.loads(self.CORS_ORIGINS)
            if isinstance(parsed, list):
                return parsed
        except Exception as e: 
            pass 
        # Fall back to comma-seperated string 
        return [s for s in self.CORS_ORIGINS.strip().split(",") if s.strip()]
        
try: 
    settings = Settings()
    try:
        with open(_debug_log_path, "a") as _f: 
            _f.write(
                json.dumps({
                    "location":"config.py:38",
                    "message": "Settings SUCCESS",
                    "data": {"cors": settings.CORS_ORIGINS}
                })
                + "\n"
            )
    except Exception:
        pass
except Exception as _e: 
    with open(_debug_log_path, "a") as _f: 
            _f.write(
                json.dumps({
                    "location":"config.py:38",
                    "message": "Settings SUCCESS",
                    "data": {"error": str(_e)}
                })
                + "\n"
            ) 
