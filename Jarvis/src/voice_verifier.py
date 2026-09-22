import os
import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier

class VoiceVerifier:
    def __init__(self, owner_embedding_path: str = "owner_voice.npy", threshold: float = 0.75):
        self.threshold = threshold
        self.owner_embedding_path = owner_embedding_path
        self.owner_embedding = None
        
        print("[SECURITY] Initializing SpeechBrain Voice Verifier...")
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )
        
        self.load_owner_embedding()

    def load_owner_embedding(self):
        if os.path.exists(self.owner_embedding_path):
            self.owner_embedding = np.load(self.owner_embedding_path)
            print(f"[SECURITY] Master voice vector loaded successfully from `{self.owner_embedding_path}`.")
        else:
            print(f"[WARNING] Master vector `{self.owner_embedding_path}` not found! Verification running in PASSTHROUGH mode.")

    def verify_audio_data(self, audio_data: np.ndarray, sample_rate: int = 16000) -> tuple[bool, float]:
        """
        Verifies incoming audio array against master embedding.
        Returns: (is_verified: bool, similarity_score: float)
        """
        if self.owner_embedding is None:
            return True, 1.0

        # Convert float32 numpy audio into PyTorch tensor [batch, samples]
        signal = torch.from_numpy(audio_data).float()
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.classifier.encode_batch(signal)
            input_embedding = embeddings.squeeze().cpu().numpy()

        # Cosine Similarity Calculation
        dot_product = np.dot(self.owner_embedding, input_embedding)
        norm_owner = np.linalg.norm(self.owner_embedding)
        norm_input = np.linalg.norm(input_embedding)
        
        similarity = dot_product / (norm_owner * norm_input + 1e-10)
        is_verified = bool(similarity >= self.threshold)

        return is_verified, float(similarity)
