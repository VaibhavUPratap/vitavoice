import os
import numpy as np
import pandas as pd
import joblib

from ml.data.base_loader import BaseDatasetLoader
from ml.preprocessing.audio import AudioPreprocessingPipeline
from ml.feature_extraction.acoustic import extract_all_acoustic_features
from ml.feature_extraction.embeddings import extract_wav2vec2_embeddings
from ml.training.synthesize_dataset import synthesize_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Column sets for the Oxford tabular dataset
# ─────────────────────────────────────────────────────────────────────────────
# Core acoustic perturbation features (classical clinical biomarkers)
OXFORD_CLINICAL_COLS = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA',
    'NHR', 'HNR',
]

# Nonlinear dynamical complexity features — proven highest discriminators for PD
# PPE and spread1 are consistently top-2 features across all published PD voice studies
OXFORD_NONLINEAR_COLS = [
    'RPDE',     # Recurrence Period Density Entropy — vocal aperiodicity complexity
    'DFA',      # Detrended Fluctuation Analysis — fractal scaling of F0
    'spread1',  # Nonlinear fundamental frequency variation (best single feature per Little 2008)
    'spread2',  # Nonlinear fundamental frequency variation 2
    'D2',       # Correlation dimension — attractor complexity of vocal fold dynamics
    'PPE',      # Pitch Period Entropy — uncertainty in pitch, top-3 PD predictor
]

ALL_OXFORD_FEATURE_COLS = OXFORD_CLINICAL_COLS + OXFORD_NONLINEAR_COLS


class OxfordDatasetLoader(BaseDatasetLoader):
    """
    Loader for the Oxford Parkinson's Disease tabular dataset.

    Operating Modes:
    ────────────────
    direct_mode=True  (default, strongly recommended):
        Reads all 22 ground-truth clinical + nonlinear features directly from
        the CSV, including the 6 high-value nonlinear complexity features
        (RPDE, DFA, spread1, spread2, D2, PPE) that are NOT reproducible from
        synthesized audio. WavLM embeddings are still extracted from synthesized
        waveforms for OOD detection and 2D visualization only.

    direct_mode=False  (legacy):
        Synthesizes voice waveforms from tabular parameters, then re-extracts
        acoustic features from the synthetic audio. This discards the nonlinear
        features and introduces synthesis bias. Preserved for backward-compat only.
    """

    def __init__(
        self,
        data_path="datasets/parkinsons.data",
        wav_dir="datasets/synthesized_wavs",
        cache_path="datasets/oxford_cache.joblib",
        direct_mode=True,
    ):
        self.data_path = data_path
        self.wav_dir = wav_dir
        self.cache_path = cache_path
        self.direct_mode = direct_mode
        self.pipeline = AudioPreprocessingPipeline()

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def load_samples(self):
        # Try loading from cache to save time
        if os.path.exists(self.cache_path):
            print(f"Loading Oxford features from cache: {self.cache_path}")
            cached_data = joblib.load(self.cache_path)
            return cached_data['X'], cached_data['y'], cached_data['metadata']

        print("Oxford cache not found. Processing raw samples...")

        if self.direct_mode:
            return self._load_direct()
        else:
            return self._load_via_synthesis()

    # ─────────────────────────────────────────────────────────────────────────
    # Mode 1 — Direct tabular extraction (recommended)
    # ─────────────────────────────────────────────────────────────────────────

    def _load_direct(self):
        """
        Reads ground-truth features directly from the Oxford CSV.

        Feature matrix structure:
          X_cli  — (N, 22) array: 16 classical acoustic + 6 nonlinear dynamic features
          X_w2v  — (N, 768) array: WavLM embeddings from synthesized audio (for viz/OOD)
          feature_names — ordered list of column names matching X_cli columns
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Missing Oxford dataset CSV at {self.data_path}")

        df = pd.read_csv(self.data_path)
        print(f"Oxford dataset loaded: {len(df)} samples, {df['status'].sum()} PD / {(df['status']==0).sum()} healthy")

        # Validate expected feature columns exist
        missing = [c for c in ALL_OXFORD_FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Oxford CSV missing expected columns: {missing}")

        # Extract all feature vectors directly from tabular data
        X_cli = df[ALL_OXFORD_FEATURE_COLS].values.astype(np.float64)
        y = df['status'].values.astype(int)
        feature_names = ALL_OXFORD_FEATURE_COLS

        print(f"Direct tabular features extracted: {X_cli.shape[1]} features "
              f"({len(OXFORD_CLINICAL_COLS)} classical + {len(OXFORD_NONLINEAR_COLS)} nonlinear)")

        # Synthesize audio and extract WavLM embeddings for visualization/OOD
        X_w2v = self._extract_wavlm_embeddings(df)

        metadata = []
        for _, row in df.iterrows():
            metadata.append({
                'name': row['name'],
                'original_status': int(row['status']),
                'is_synthetic': False,  # Features come from ground-truth clinical measurements
                'source': 'oxford_direct'
            })

        X_dict = {
            'X_cli': X_cli,
            'X_w2v': X_w2v,
            'feature_names': feature_names
        }

        joblib.dump({'X': X_dict, 'y': y, 'metadata': metadata}, self.cache_path)
        print(f"Cached Oxford direct-mode features to {self.cache_path}")

        return X_dict, y, metadata

    def _extract_wavlm_embeddings(self, df):
        """
        Synthesizes voice samples (for WavLM embedding extraction only).
        Returns (N, 768) float32 array.
        """
        from ml.training.synthesize_dataset import synthesize_dataset

        if not os.path.exists(self.wav_dir) or len(os.listdir(self.wav_dir)) < 190:
            print("[INFO] Synthesizing audio for WavLM embeddings (visualization/OOD only)...")
            synthesize_dataset(self.data_path, self.wav_dir)

        embeddings = []
        for _, row in df.iterrows():
            wav_path = os.path.join(self.wav_dir, f"{row['name']}.wav")
            try:
                y_audio, sr = self.pipeline.preprocess_audio(wav_path)
                emb = extract_wav2vec2_embeddings(y_audio, sr)
                embeddings.append(emb)
            except Exception as e:
                print(f"[WARN] WavLM embedding failed for {row['name']}: {e}. Using zeros.")
                embeddings.append(np.zeros(768, dtype=np.float32))

        return np.array(embeddings, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Mode 2 — Legacy synthesis-based extraction
    # ─────────────────────────────────────────────────────────────────────────

    def _load_via_synthesis(self):
        """
        Legacy mode: synthesizes audio from tabular params and re-extracts features.
        NOTE: This discards the nonlinear features (RPDE, DFA, PPE, etc.) and
        introduces synthesis bias. Use direct_mode=True instead.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Missing Oxford dataset CSV at {self.data_path}")

        if not os.path.exists(self.wav_dir) or len(os.listdir(self.wav_dir)) < 190:
            print("[WARNING] Synthesizing representative vocal waveforms from tabular metrics...")
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
                y_audio, sr = self.pipeline.preprocess_audio(wav_path)
                cli_feats = extract_all_acoustic_features(y_audio, sr)
                if sorted_keys is None:
                    sorted_keys = sorted(cli_feats.keys())
                cli_vec = np.array([cli_feats[k] for k in sorted_keys])
                w2v_emb = extract_wav2vec2_embeddings(y_audio, sr)

                clinical_features.append(cli_vec)
                w2v_embeddings.append(w2v_emb)
                labels.append(status)
                metadata.append({
                    'name': name,
                    'original_status': status,
                    'is_synthetic': True,
                    'source': 'oxford_synthesis'
                })
            except Exception as e:
                print(f"Error loading sample {name}: {e}")

        X_cli = np.array(clinical_features)
        X_w2v = np.array(w2v_embeddings)
        y = np.array(labels)

        X_dict = {
            'X_cli': X_cli,
            'X_w2v': X_w2v,
            'feature_names': sorted_keys
        }

        joblib.dump({'X': X_dict, 'y': y, 'metadata': metadata}, self.cache_path)
        return X_dict, y, metadata
