import os
import json
import torch
from tqdm import tqdm
from jiwer import wer
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
)
import whisper
from shared.models import DeepSpeech2
from src.config import LIBRISPEECH_DIR as DATA_DIR, SAVED_MODELS_DIR
from src.utils.deepspeech2_utils import load_audio, audio_to_mel
from src.utils.text_utils import normalize

SAMPLING_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_test_samples(split="dev-clean", max_samples=200):
    """Return list of (audio_path, transcript) from a LibriSpeech split."""
    samples = []
    split_dir = DATA_DIR / split
    for speaker in os.listdir(split_dir):
        for section in os.listdir(split_dir / speaker):
            section_dir = split_dir / speaker / section
            txt_files = list(section_dir.glob("*.txt"))
            if not txt_files:
                continue
            with open(txt_files[0]) as f:
                for line in f:
                    parts = line.split()
                    audio_path = section_dir / (parts[0] + ".flac")
                    transcript = " ".join(parts[1:]).strip()
                    samples.append((str(audio_path), transcript))
                    if len(samples) >= max_samples:
                        return samples
    return samples

# ── model inference ───────────────────────────────────────────────────────────

def transcribe_deepspeech2(audio, model, tokenizer):
    mel = audio_to_mel(audio)
    mel = mel.unsqueeze(0)
    src_len = torch.tensor([mel.shape[-1]])
    with torch.no_grad():
        logits, _ = model(mel.to(DEVICE), src_len)
    tokens = logits.squeeze().argmax(dim=-1).tolist()
    return tokenizer.decode(tokens)


def transcribe_wav2vec2(audio, model, processor):
    inputs = processor(
        audio.squeeze().numpy(),
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(ids)[0]


def transcribe_whisper(audio_path, model):
    result = model.transcribe(audio_path, language="en")
    return result["text"].strip()


# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate(model_name, transcribe_fn, samples):
    hypotheses, references = [], []
    for audio_path, transcript in tqdm(samples, desc=model_name):
        try:
            audio = load_audio(audio_path)
            pred = transcribe_fn(audio, audio_path)
            hypotheses.append(normalize(pred))
            references.append(normalize(transcript))
        except Exception as e:
            print(f"  skipped {audio_path}: {e}")
    score = wer(references, hypotheses)
    print(f"\n{model_name} WER: {score:.4f} ({score*100:.2f}%)\n")
    return score


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    samples = load_test_samples(split="dev-clean", max_samples=200)
    print(f"Loaded {len(samples)} samples from dev-clean\n")

    results = {}

    # ── DeepSpeech2 ──
    config_path = SAVED_MODELS_DIR / "config.json"
    weights_path = SAVED_MODELS_DIR / "best_weights.pt"
    with open(config_path) as f:
        config = json.load(f)

    ds2_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
    ds2_model = DeepSpeech2(
        rnn_hidden_size=config["rnn_hidden_size"],
        rnn_depth=config["rnn_depth"],
    ).to(DEVICE)
    ds2_model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    ds2_model.eval()

    results["DeepSpeech2"] = evaluate(
        "DeepSpeech2",
        lambda audio, _: transcribe_deepspeech2(audio, ds2_model, ds2_tokenizer),
        samples,
    )

    # ── Wav2Vec2 ──
    w2v_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    w2v_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
    w2v_model.eval()

    results["Wav2Vec2"] = evaluate(
        "Wav2Vec2",
        lambda audio, _: transcribe_wav2vec2(audio, w2v_model, w2v_processor),
        samples,
    )

    # ── Whisper ──
    whisper_model = whisper.load_model("tiny", device=DEVICE)

    results["Whisper"] = evaluate(
        "Whisper",
        lambda _, path: transcribe_whisper(path, whisper_model),
        samples,
    )

    # ── summary ──
    print("=" * 40)
    print(f"{'Model':<15} {'WER':>10}")
    print("-" * 40)
    for name, score in results.items():
        print(f"{name:<15} {score*100:>9.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
