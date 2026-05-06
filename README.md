# Speech-to-Text Recognition System

A full-stack web application for audio transcription using multiple deep learning models. Users can upload audio files or record directly in the browser and receive real-time transcriptions.

## Models

| Model | Type | Source |
|-------|------|--------|
| DeepSpeech2 | Custom-trained (Conv2D + RNN + CTC) | Trained on LibriSpeech |
| Wav2Vec2 | Pre-trained | `facebook/wav2vec2-base-960h` |
| Whisper | Pre-trained | OpenAI `tiny` |

## Architecture

```
sttproject/
├── ml/          # Training pipeline, datasets, notebooks
├── backend/     # FastAPI REST API + inference service
├── frontend/    # React web app
└── shared/      # Shared model definitions and utilities
```

## ML Pipeline

- **Dataset**: LibriSpeech (`train-clean-100`, `train-clean-360`, `dev-clean`)
- **Preprocessing**: 16 kHz resampling → mono → VAD → Mel-spectrogram (80 bins) → Z-score normalization
- **Training**: AdamW optimizer, cosine annealing with warmup, CTC loss, batch size 32
- **Evaluation**: Word Error Rate (WER) via `jiwer`

## Tech Stack

- **ML**: PyTorch, torchaudio, HuggingFace Transformers
- **Backend**: FastAPI, SQLAlchemy, JWT authentication, aiosmtplib
- **Frontend**: React, Material-UI, MediaRecorder API

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm start
```

### Training

```bash
cd ml
pip install -r requirements.txt
python src/trainer.py
```

Environment variables (create `backend/.env`):

```env
DATABASE_URL=sqlite:///./app/db/stt.db
JWT_SECRET_KEY=your-secret-key
CORS_ORIGINS=http://localhost:3000
DEEPSPEECH2_MODEL_PATH=../ml/saved_models/best_weights.pt
DEEPSPEECH2_CONFIG_PATH=../ml/saved_models/config.json
WHISPER_MODEL_NAME=tiny
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Login, returns JWT |
| POST | `/api/stt/predict` | Transcribe audio file |

`POST /api/stt/predict` accepts `multipart/form-data`:
- `file` — audio file (max 5MB)
- `language` — language code or `au` for auto-detect
- `model` — `deepspeech2`, `wav2vec2`, `wav2vec2-finnish`, or `whisper`
