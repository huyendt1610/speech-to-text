from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = ML_DIR.parent


DATA_DIR = ML_DIR / "data"
LIBRISPEECH_DIR = DATA_DIR / "LibriSpeech"
DATASET_CACHE_DIR = DATA_DIR / "dataset_cache"


CONFIGS_DIR = ML_DIR / "configs"
OUTPUTS_DIR = ML_DIR / "outputs"

SAVED_MODELS_DIR = OUTPUTS_DIR/ "saved_models"
