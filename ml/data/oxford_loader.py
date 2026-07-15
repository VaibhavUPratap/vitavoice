import os
import numpy as np
import pandas as pd
import joblib

from ml.data.base_loader import BaseDatasetLoader
from ml.preprocessing.audio import AudioPreprocessingPipeline
from ml.feature_extraction.acoustic import extract_all_acoustic_features
from ml.feature_extraction.embeddings import extract_wav2vec2_embeddings
from ml.training.synthesize_dataset import synthesize_dataset

class OxfordDatasetLoader(BaseDatasetLoader):
    """
    Loader for the Oxford Parkinson's Disease tabular dataset.
    Bridges clinical metrics with demo synthetic speech representations.
    """
    
    def __init__(self, data_path="datasets/parkinsons.data", wav_dir="datasets/synthesized_wavs", cache_path="datasets/oxford_cache.joblib"):
        self.data_path = data_path
        self.wav_dir = wav_dir
        self.cache_path = cache_path
        self.pipeline = AudioPreprocessingPipeline()
        
    def load_samples(self):
        # 1. Try loading from cache to save time
        if os.path.exists(self.cache_path):
            print(f"Loading Oxford features from cache: {self.cache_path}")
            cached_data = joblib.load(self.cache_path)
            return cached_data['X'], cached_data['y'], cached_data['metadata']
            
        print("Oxford cache not found. Processing raw samples...")
        
        # 2. Check if dataset CSV exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Missing Oxford dataset CSV at {self.data_path}")
            
        # Check if synthesized waveforms exist, if not trigger synthetic fallback
        if not os.path.exists(self.wav_dir) or len(os.listdir(self.wav_dir)) < 190:
            print("[WARNING] Genuine voice recordings for the Oxford dataset are not available in this workspace.")
            print("[WARNING] VitaVoice will synthesize representative vocal waveforms from tabular metrics for prototype demonstration.")
            synthesize_dataset(self.data_path, self.wav_dir)
            
        df = pd.read_csv(self.data_path)
        
        clinical_features = []
        w2v_embeddings = []
        labels = []
        metadata = []
        
        sorted_keys = None
        
        for idx, row in df.iterrows():
            name = row['name']
            status = row['status']
            wav_path = os.path.join(self.wav_dir, f"{name}.wav")
            
            if not os.path.exists(wav_path):
                continue
                
            try:
                # Preprocess audio using the modular pipeline
                y, sr = self.pipeline.preprocess_audio(wav_path)
                
                # Extract clinical features
                cli_feats = extract_all_acoustic_features(y, sr)
                if sorted_keys is None:
                    sorted_keys = sorted(cli_feats.keys())
                
                cli_vec = np.array([cli_feats[k] for k in sorted_keys])
                
                # Extract WavLM Base embeddings
                w2v_emb = extract_wav2vec2_embeddings(y, sr)
                
                clinical_features.append(cli_vec)
                w2v_embeddings.append(w2v_emb)
                labels.append(status)
                
                # Record metadata for trace analysis
                metadata.append({
                    'name': name,
                    'original_status': status,
                    'is_synthetic': True  # Marks this as synthesized demo data
                })
            except Exception as e:
                print(f"Error loading sample {name}: {e}")
                
        X_cli = np.array(clinical_features)
        X_w2v = np.array(w2v_embeddings)
        y = np.array(labels)
        
        # Combine clinical features and full WavLM embeddings
        # (We will apply PCA to the WavLM component during training)
        X_dict = {
            'X_cli': X_cli,
            'X_w2v': X_w2v,
            'feature_names': sorted_keys
        }
        
        # Cache results
        joblib.dump({
            'X': X_dict,
            'y': y,
            'metadata': metadata
        }, self.cache_path)
        
        return X_dict, y, metadata
