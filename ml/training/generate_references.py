import os
import sys
import numpy as np
import joblib
from sklearn.svm import OneClassSVM
from sklearn.covariance import LedoitWolf

def generate_wavlm_references(cache_path="datasets/oxford_cache.joblib", checkpoints_dir="ml/checkpoints"):
    print(f"Loading cached dataset from {cache_path}...")
    if not os.path.exists(cache_path):
        print(f"Error: dataset cache not found at {cache_path}")
        return False
        
    cache = joblib.load(cache_path)
    X_dict = cache['X']
    y_arr = np.array(cache['y'])
    
    X_w2v = X_dict['X_w2v']
    print(f"Loaded WavLM embeddings shape: {X_w2v.shape}")
    
    # 1. Compute Centroids
    healthy_mask = (y_arr == 0)
    parkinsons_mask = (y_arr == 1)
    
    healthy_centroid = np.mean(X_w2v[healthy_mask], axis=0)
    parkinsons_centroid = np.mean(X_w2v[parkinsons_mask], axis=0)
    print("Computed Healthy and Parkinson's centroids.")
    
    # 2. Fit OOD Detector (OneClassSVM)
    print("Fitting One-Class SVM OOD detector...")
    ood_detector = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
    ood_detector.fit(X_w2v)
    
    # Calculate scores on training data to establish threshold
    train_scores = ood_detector.score_samples(X_w2v)
    ood_threshold = float(np.percentile(train_scores, 5))
    print(f"OOD threshold established (5th percentile): {ood_threshold:.4f}")
    
    # 3. Compute Covariance Matrices for Mahalanobis Distance in 16D
    reducer_path = os.path.join(checkpoints_dir, "reducer.joblib")
    if os.path.exists(reducer_path):
        print(f"Loading PCA reducer from {reducer_path}...")
        reducer = joblib.load(reducer_path)
        if reducer is not None:
            X_w2v_reduced = reducer.transform(X_w2v)
        else:
            X_w2v_reduced = X_w2v
    else:
        print("Warning: reducer.joblib not found. Using raw embeddings for covariance.")
        X_w2v_reduced = X_w2v
        
    print(f"Reduced embeddings shape for Mahalanobis: {X_w2v_reduced.shape}")
    
    mean_healthy_reduced = np.mean(X_w2v_reduced[healthy_mask], axis=0)
    mean_parkinsons_reduced = np.mean(X_w2v_reduced[parkinsons_mask], axis=0)
    
    # Estimate covariance with Ledoit-Wolf shrinkage to ensure well-conditioned inverse
    cov_healthy = LedoitWolf().fit(X_w2v_reduced[healthy_mask]).covariance_
    cov_parkinsons = LedoitWolf().fit(X_w2v_reduced[parkinsons_mask]).covariance_
    
    # Add minor regularization to diagonal for extreme robustness
    reg = 1e-4 * np.eye(X_w2v_reduced.shape[1])
    cov_healthy_inv = np.linalg.inv(cov_healthy + reg)
    cov_parkinsons_inv = np.linalg.inv(cov_parkinsons + reg)
    print("Computed covariance inverses with Ledoit-Wolf shrinkage and regularization.")
    
    # 4. Save references
    os.makedirs(checkpoints_dir, exist_ok=True)
    out_path = os.path.join(checkpoints_dir, "wavlm_references.joblib")
    references = {
        'healthy_centroid': healthy_centroid,
        'parkinsons_centroid': parkinsons_centroid,
        'train_embeddings': X_w2v,
        'train_labels': y_arr,
        'ood_detector': ood_detector,
        'ood_threshold': ood_threshold,
        'mean_healthy_reduced': mean_healthy_reduced,
        'mean_parkinsons_reduced': mean_parkinsons_reduced,
        'cov_healthy_inv': cov_healthy_inv,
        'cov_parkinsons_inv': cov_parkinsons_inv
    }
    
    joblib.dump(references, out_path)
    print(f"Successfully saved WavLM references to {out_path}")
    return True

if __name__ == "__main__":
    generate_wavlm_references()
