import os
import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

class VoiceVerifier:
    def __init__(self, owner_embedding_path: str = "owner_voice.npy", threshold: float = 0.75):
        self.threshold = threshold
        self.owner_embedding_path = owner_embedding_path
        self.owner_embedding = None
        
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        )
        
        self.load_owner_embedding()

    def load_owner_embedding(self):
        if os.path.exists(self.owner_embedding_path):
            self.owner_embedding = np.load(self.owner_embedding_path)
            print(f"[SECURITY] Owner embedding loaded from {self.owner_embedding_path}.")
        else:
            print(f"[WARNING] {self.owner_embedding_path} not found. Voice verification running in passthrough mode.")

    def verify_audio_tensor(self, audio_tensor: torch.Tensor, sample_rate: int = 16000) -> tuple[bool, float]:
        if self.owner_embedding is None:
            return True, 1.0

        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.classifier.encode_batch(audio_tensor)
            input_embedding = embeddings.squeeze().cpu().numpy()

        dot_product = np.dot(self.owner_embedding, input_embedding)
        norm_owner = np.linalg.norm(self.owner_embedding)
        norm_input = np.linalg.norm(input_embedding)
        
        similarity = dot_product / (norm_owner * norm_input)
        is_verified = bool(similarity >= self.threshold)

        return is_verified, float(similarity)