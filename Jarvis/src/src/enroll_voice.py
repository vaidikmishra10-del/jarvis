import os
import wave
import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

# Load SpeechBrain ECAPA-TDNN pre-trained speaker recognition model
print("Loading SpeechBrain speaker recognition model...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
)

OUTPUT_WAV = "owner_voice_sample.wav"
OUTPUT_NPY = "owner_voice.npy"

def record_audio(filename, duration=15, samplerate=16000):
    """Record audio using sounddevice or simple wave prompt instructions."""
    try:
        import sounddevice as sd
        print(f"\n[RECORDING] Please speak continuously for {duration} seconds...")
        print("Suggested phrase: 'I am Jarvis's primary user. Enable secure voice authentication protocols.'")
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        print("[RECORDING COMPLETE]")
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(recording.tobytes())
    except ImportError:
        print("Please install sounddevice (`pip install sounddevice`) or provide a 16kHz WAV file named `owner_voice_sample.wav`.")

def extract_and_save_embedding():
    if not os.path.exists(OUTPUT_WAV):
        record_audio(OUTPUT_WAV)

    print("\nExtracting speaker vector embedding...")
    signal, fs = torchaudio.load(OUTPUT_WAV)
    
    # Ensure mono channel
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
        
    embeddings = classifier.encode_batch(signal)
    master_embedding = embeddings.squeeze().cpu().numpy()

    np.save(OUTPUT_NPY, master_embedding)
    print(f"Master voice embedding saved successfully to `{OUTPUT_NPY}`! Vector shape: {master_embedding.shape}")

if __name__ == "__main__":
    extract_and_save_embedding()    