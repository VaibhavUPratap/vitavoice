import os
import numpy as np
import joblib

from ml.data.base_loader import BaseDatasetLoader
from ml.preprocessing.audio import AudioPreprocessingPipeline
from ml.feature_extraction.acoustic import extract_all_acoustic_features
from ml.feature_extraction.embeddings import extract_wav2vec2_embeddings

class RealAudioDatasetLoader(BaseDatasetLoader):
    """
    Loader for genuine clinical voice datasets.
    Scans folders containing actual audio WAV/MP3 files.
    """
    
    def __init__(self, root_dir="datasets/real", cache_path="datasets/real_cache.joblib"):
        self.root_dir = root_dir
        self.cache_path = cache_path
        self.pipeline = AudioPreprocessingPipeline()
        
    def load_samples(self):
        # 1. Try loading from cache
        if os.path.exists(self.cache_path):
            print(f"Loading real voice features from cache: {self.cache_path}")
            cached_data = joblib.load(self.cache_path)
            return cached_data['X'], cached_data['y'], cached_data['metadata']
            
        print(f"Scanning directory: {self.root_dir} for raw recordings...")
        
        healthy_dir = os.path.join(self.root_dir, "healthy")
        parkinsons_dir = os.path.join(self.root_dir, "parkinsons")
        
        # Ensure folders exist
        os.makedirs(healthy_dir, exist_ok=True)
        os.makedirs(parkinsons_dir, exist_ok=True)
        
        audio_files = []
        
        # Check files in healthy directory (label = 0)
        for f in os.listdir(healthy_dir):
            if f.endswith(('.wav', '.mp3')):
                audio_files.append((os.path.join(healthy_dir, f), 0, f))
                
        # Check files in Parkinson's directory (label = 1)
        for f in os.listdir(parkinsons_dir):
            if f.endswith(('.wav', '.mp3')):
                audio_files.append((os.path.join(parkinsons_dir, f), 1, f))
                
        if len(audio_files) == 0:
            print(f"[INFO] No genuine audio recordings found in '{self.root_dir}'.")
            print("[INFO] Please place raw patient .wav files in 'healthy/' and 'parkinsons/' directories.")
            return {
                'X_cli': np.zeros((0, 33)),
                'X_w2v': np.zeros((0, 768)),
                'feature_names': []
            }, np.array([]), []
            
        clinical_features = []
        w2v_embeddings = []
        labels = []
        metadata = []
        
        sorted_keys = None
        
        for file_path, label, filename in audio_files:
            try:
                # Run the preprocessing pipeline
                y, sr = self.pipeline.preprocess_audio(file_path)
                
                # Extract clinical metrics
                cli_feats = extract_all_acoustic_features(y, sr)
                if sorted_keys is None:
                    sorted_keys = sorted(cli_feats.keys())
                    
                cli_vec = np.array([cli_feats[k] for k in sorted_keys])
                
                # Extract WavLM Base representations
                w2v_emb = extract_wav2vec2_embeddings(y, sr)
                
                clinical_features.append(cli_vec)
                w2v_embeddings.append(w2v_emb)
                labels.append(label)
                
                metadata.append({
                    'name': filename,
                    'original_status': label,
                    'is_synthetic': False  # Mark as real clinical recording
                })
            except Exception as e:
                print(f"Error loading real sample {filename}: {e}")
                
        X_cli = np.array(clinical_features)
        X_w2v = np.array(w2v_embeddings)
        y = np.array(labels)
        
        X_dict = {
            'X_cli': X_cli,
            'X_w2v': X_w2v,
            'feature_names': sorted_keys
        }
        
        # Cache results if we actually loaded files
        if len(audio_files) > 0:
            joblib.dump({
                'X': X_dict,
                'y': y,
                'metadata': metadata
            }, self.cache_path)
            
        return X_dict, y, metadata
