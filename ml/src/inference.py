import torch
from src.utils.deepspeech2_utils import load_audio, audio_to_mel

def inference(path_to_audio, model, tokenizer, sampling_rate=16000, device="cpu"):
    audio = load_audio(path_to_audio, sampling_rate)
    return _decode(audio, model, tokenizer, sampling_rate, device)

def inference2(audio, model, tokenizer, sampling_rate=16000, device="cpu"):
    return _decode(audio, model, tokenizer, sampling_rate, device)

def _decode(audio, model, tokenizer, sampling_rate, device):
    mel = audio_to_mel(audio, sampling_rate).unsqueeze(0)
    src_len = torch.tensor([mel.shape[-1]])
    model = model.to(device)
    with torch.no_grad():
        pred_logits, _ = model(mel.to(device), src_len)
    pred_tokens = pred_logits.squeeze().argmax(axis=-1).tolist()
    return tokenizer.decode(pred_tokens)

if __name__ == "__main__":
    print("hello")
