import torch 
import torchaudio
import torchaudio.transforms as T 
import io

def inference(path_to_audio, model, tokenizer, sampling_rate = 16000, device="cpu"): 

    # Transforms
    audio2mels = T.MelSpectrogram(
                sample_rate = sampling_rate,
                n_mels=80 
            )

    amp2db = T.AmplitudeToDB(
        top_db=80.0
    ) 

    audio, orig_sr = torchaudio.load(path_to_audio, normalize=True) #normalize: true, convert into [-1, 1], normalize: false, audio bit is between [0, 255]
    if orig_sr != sampling_rate:
        audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=sampling_rate) # re-sample to 16.000

    # audio: waveform
    mel = audio2mels(audio) # to MelSpectrogram
    # print(mel.shape)
    # print(audio.shape) => [1, 170400]
    # print(mel.shape) => [1, 80, 853]: 80 bins, 
    # 853 time steps: after sliding window of hann_window (400 windowsize with 200 overlap default value)

    mel = amp2db(mel) # to amplitude to Decibel => see diff frequencies lighting up at the diff time steps
    # print(mel.shape)

    # plt.figure(figsize=(15,5))
    # plt.imshow(mel[0])
    # plt.show()

    mel = (mel - mel.mean())/(mel.std() + 1e-6) # 1e-6 to avoid deviding by zero, to nomalize
    mel = mel.unsqueeze(0) # add in batch dimension 
    # print(mel.shape)

    src_len = torch.tensor([mel.shape[-1]]) # compute src_len

    model = model.to(device) # set model device 

    with torch.no_grad():
        pred_logits, _ = model(mel.to(device), src_len)

    pred_tokens = pred_logits.squeeze().argmax(axis=-1).tolist() 

    pred_transcript = tokenizer.decode(pred_tokens)

    return pred_transcript

def inference2(audio_bytes, model, tokenizer, sampling_rate = 16000, device="cpu"): 

    audio, orig_sr = torchaudio.load(io.BytesIO(audio_bytes), normalize=True) #normalize: true, convert into [-1, 1], normalize: false, audio bit is between [0, 255]
    if orig_sr != sampling_rate:
        audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=sampling_rate) # re-sample to 16.000

    # Transforms
    audio2mels = T.MelSpectrogram(
                sample_rate = sampling_rate,
                n_mels=80 
            )

    amp2db = T.AmplitudeToDB(
        top_db=80.0
    ) 
    # audio: waveform
    mel = audio2mels(audio) 
    mel = amp2db(mel) 
    mel = (mel - mel.mean())/(mel.std() + 1e-6) # 1e-6 to avoid deviding by zero, to nomalize
    mel = mel.unsqueeze(0) # add in batch dimension 
    # print(mel.shape)

    src_len = torch.tensor([mel.shape[-1]]) # compute src_len

    model = model.to(device) # set model device 

    with torch.no_grad():
        pred_logits, _ = model(mel.to(device), src_len)

    pred_tokens = pred_logits.squeeze().argmax(axis=-1).tolist() 

    pred_transcript = tokenizer.decode(pred_tokens)

    return pred_transcript

if __name__ == "__main__": 
    print("hello")