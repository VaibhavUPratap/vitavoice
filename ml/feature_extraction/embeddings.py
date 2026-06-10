import os
import numpy as np
import torch
import librosa
import joblib
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Use CPU by default for inference to avoid GPU memory issues in development
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache model and processor globally so they aren't reloaded on every call
_processor = None
_model = None

def get_wav2vec2_resources():
    """
    Lazy loads and caches the Wav2Vec2 model and processor.
    """
    global _processor, _model
    if _processor is None or _model is None:
        model_name = "facebook/wav2vec2-base-960h"
        # Download and load processor and model
        _processor = Wav2Vec2Processor.from_pretrained(model_name)
        _model = Wav2Vec2Model.from_pretrained(model_name)
        _model.to(device)
        _model.eval()
    return _processor, _model

def extract_wav2vec2_embeddings(y, sr):
    """
    Extracts the 768-dimensional embedding from the preprocessed audio waveform.
    """
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
        
    processor, model = get_wav2vec2_resources()
    
    # Process audio
    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract last hidden state, shape [batch_size, sequence_length, 768]
        last_hidden_state = outputs.last_hidden_state
        
        # Mean pool over sequence length to get utterance-level embedding, shape [768]
        embedding = torch.mean(last_hidden_state, dim=1).squeeze().cpu().numpy()
        
    return embedding

def load_pca_projection(pca_path="ml/checkpoints/pca_model.joblib"):
    """
    Loads the trained PCA model for 2D embedding space visualization.
    """
    if os.path.exists(pca_path):
        return joblib.load(pca_path)
    return None

def project_embedding_2d(embedding, pca_model=None):
    """
    Projects a 768-dimensional embedding into a 2D coordinate using PCA.
    If no PCA model is loaded, generates a default coordinate based on feature heuristics.
    """
    if pca_model is None:
        pca_model = load_pca_projection()
        
    if pca_model is not None:
        # Project using loaded PCA
        coords = pca_model.transform(embedding.reshape(1, -1))[0]
        return float(coords[0]), float(coords[1])
    else:
        # Heuristic projection fallback: maps general amplitude and pitch characteristics
        # to arbitrary but deterministic coordinates to draw clusters.
        # This prevents crashes if PCA is not trained.
        norm_val = np.linalg.norm(embedding)
        x = float(np.dot(embedding[:384], np.ones(384)) / norm_val) * 10
        y = float(np.dot(embedding[384:], np.ones(384)) / norm_val) * 10
        return x, y
