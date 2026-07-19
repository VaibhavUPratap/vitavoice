import os
import joblib
import numpy as np

class WavLMFingerprintEngine:
    def __init__(self, checkpoints_dir="ml/checkpoints"):
        self.checkpoints_dir = checkpoints_dir
        self.loaded = False
        self.healthy_centroid = None
        self.parkinsons_centroid = None
        self.train_embeddings = None
        self.train_labels = None
        self.mean_healthy_reduced = None
        self.mean_parkinsons_reduced = None
        self.cov_healthy_inv = None
        self.cov_parkinsons_inv = None
        self.reducer = None
        self.load_references()

    def load_references(self):
        refs_path = os.path.join(self.checkpoints_dir, "wavlm_references.joblib")
        reducer_path = os.path.join(self.checkpoints_dir, "reducer.joblib")
        
        if os.path.exists(refs_path):
            try:
                refs = joblib.load(refs_path)
                self.healthy_centroid = refs.get('healthy_centroid')
                self.parkinsons_centroid = refs.get('parkinsons_centroid')
                self.train_embeddings = refs.get('train_embeddings')
                self.train_labels = refs.get('train_labels')
                self.mean_healthy_reduced = refs.get('mean_healthy_reduced')
                self.mean_parkinsons_reduced = refs.get('mean_parkinsons_reduced')
                self.cov_healthy_inv = refs.get('cov_healthy_inv')
                self.cov_parkinsons_inv = refs.get('cov_parkinsons_inv')
                self.loaded = True
            except Exception as e:
                print(f"Error loading WavLM references: {e}")
                self._setup_dummy_references()
        else:
            print(f"Warning: References file not found at {refs_path}. Initializing dummies.")
            self._setup_dummy_references()
            
        if os.path.exists(reducer_path):
            try:
                self.reducer = joblib.load(reducer_path)
            except Exception as e:
                print(f"Error loading PCA/UMAP reducer: {e}")

    def _setup_dummy_references(self):
        self.healthy_centroid = np.zeros(768)
        self.parkinsons_centroid = np.zeros(768)
        self.train_embeddings = np.zeros((10, 768))
        self.train_labels = np.zeros(10)
        self.mean_healthy_reduced = np.zeros(16)
        self.mean_parkinsons_reduced = np.zeros(16)
        self.cov_healthy_inv = np.eye(16)
        self.cov_parkinsons_inv = np.eye(16)
        self.loaded = False

    def compute_similarity(self, embedding: np.ndarray) -> dict:
        """
        Computes cosine similarities, Mahalanobis distances, and normalized similarity index.
        """
        norm_emb = embedding / (np.linalg.norm(embedding) + 1e-10)
        norm_h = self.healthy_centroid / (np.linalg.norm(self.healthy_centroid) + 1e-10)
        norm_p = self.parkinsons_centroid / (np.linalg.norm(self.parkinsons_centroid) + 1e-10)
        
        sim_healthy = float(np.dot(norm_emb, norm_h))
        sim_parkinsons = float(np.dot(norm_emb, norm_p))
        
        # Softmax temperature normalization to derive prototype confidence (Temp = 0.05)
        temp = 0.05
        exp_p = np.exp(sim_parkinsons / temp)
        exp_h = np.exp(sim_healthy / temp)
        similarity_score_pd = float(exp_p / (exp_p + exp_h)) * 100.0
        similarity_score_healthy = 100.0 - similarity_score_pd
        
        nearest_cluster = "Parkinson's Cluster" if sim_parkinsons > sim_healthy else "Healthy Cluster"
        
        # Determine confidence of cluster association based on spacing margin
        sim_diff = abs(sim_parkinsons - sim_healthy)
        if sim_diff >= 0.04:
            embedding_confidence = "High"
        elif sim_diff >= 0.015:
            embedding_confidence = "Moderate"
        else:
            embedding_confidence = "Low"
            
        # Calculate Mahalanobis distances in 16D space
        d_mahalanobis_healthy = 0.0
        d_mahalanobis_parkinsons = 0.0
        
        if self.reducer is not None:
            try:
                e_reduced = self.reducer.transform(embedding.reshape(1, -1))[0]
                
                diff_h = e_reduced - self.mean_healthy_reduced
                d_mahalanobis_healthy = float(np.sqrt(np.dot(np.dot(diff_h, self.cov_healthy_inv), diff_h.T)))
                
                diff_p = e_reduced - self.mean_parkinsons_reduced
                d_mahalanobis_parkinsons = float(np.sqrt(np.dot(np.dot(diff_p, self.cov_parkinsons_inv), diff_p.T)))
            except Exception as e:
                print(f"Error calculating Mahalanobis distances: {e}")
                
        return {
            "similarity_healthy": round(similarity_score_healthy, 1),
            "similarity_parkinsons": round(similarity_score_pd, 1),
            "sim_healthy_cosine": round(sim_healthy, 4),
            "sim_parkinsons_cosine": round(sim_parkinsons, 4),
            "nearest_cluster": nearest_cluster,
            "embedding_confidence": embedding_confidence,
            "mahalanobis_healthy": round(d_mahalanobis_healthy, 2),
            "mahalanobis_parkinsons": round(d_mahalanobis_parkinsons, 2)
        }

    def find_nearest_neighbors(self, embedding: np.ndarray, k: int = 5) -> list:
        """
        Retrieves the K nearest reference recordings based on cosine similarity.
        """
        if self.train_embeddings is None or len(self.train_embeddings) == 0:
            return []
            
        norm_emb = embedding / (np.linalg.norm(embedding) + 1e-10)
        norm_train = self.train_embeddings / (np.linalg.norm(self.train_embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Calculate similarities to all reference nodes
        similarities = np.dot(norm_train, norm_emb)
        
        # Get top-K indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            label = int(self.train_labels[idx])
            cohort_name = "Pathology Cohort (PD)" if label == 1 else "Healthy Control"
            results.append({
                "reference_index": int(idx),
                "similarity": round(float(similarities[idx]), 4),
                "cohort": cohort_name,
                "label": label
            })
        return results
