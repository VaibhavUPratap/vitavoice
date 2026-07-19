import os
import argparse
import json
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Local dataset loaders
from ml.data.oxford_loader import OxfordDatasetLoader
from ml.data.real_audio_loader import RealAudioDatasetLoader

def train_vita_voice(dataset_type="oxford", root_dir="datasets", reduction_method="pca", n_components=16, checkpoints_dir="ml/checkpoints"):
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # 1. Instantiate the selected Dataset Loader
    if dataset_type == "oxford":
        data_path = os.path.join(root_dir, "parkinsons.data")
        wav_dir = os.path.join(root_dir, "synthesized_wavs")
        cache_path = os.path.join(root_dir, "oxford_cache.joblib")
        # Oxford loader has cache, so we can delete the cache to force reload if features shape changed
        # Since we added new clinical features, we should force delete the cache
        if os.path.exists(cache_path):
            os.remove(cache_path)
        print(f"Loading Oxford demo dataset from {data_path}...")
        loader = OxfordDatasetLoader(data_path=data_path, wav_dir=wav_dir, cache_path=cache_path)
    elif dataset_type == "real":
        real_dir = os.path.join(root_dir, "real")
        cache_path = os.path.join(root_dir, "real_cache.joblib")
        if os.path.exists(cache_path):
            os.remove(cache_path)
        print(f"Loading real clinical voice recordings from {real_dir}...")
        loader = RealAudioDatasetLoader(root_dir=real_dir, cache_path=cache_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
        
    X_dict, y_arr, metadata = loader.load_samples()
    
    if len(y_arr) == 0:
        print("[ERROR] Cannot train model. The selected dataset is empty.")
        return False
        
    X_cli = X_dict['X_cli']
    X_w2v = X_dict['X_w2v']
    feature_names = X_dict['feature_names']
    
    # Write feature names index for reference during inference
    joblib.dump(feature_names, os.path.join(checkpoints_dir, "feature_names.joblib"))
    
    # 2. PCA for 2D visualization (always PCA, 2 components for clusters plot)
    print("Fitting PCA for 2D visualization clusters...")
    pca_vis = PCA(n_components=2)
    X_w2v_2d = pca_vis.fit_transform(X_w2v)
    joblib.dump(pca_vis, os.path.join(checkpoints_dir, "pca_model.joblib"))
    
    # Save coordinate references
    import pandas as pd
    vis_df = pd.DataFrame({
        'pca_x': X_w2v_2d[:, 0],
        'pca_y': X_w2v_2d[:, 1],
        'status': y_arr
    })
    vis_df.to_csv("datasets/pca_visualization_clusters.csv", index=False)
    
    # 3. Dimensionality Reduction Layer
    if reduction_method == "pca":
        print(f"Fitting PCA reduction for neural embeddings ({n_components} components)...")
        reducer = PCA(n_components=n_components, random_state=42)
        X_w2v_reduced = reducer.fit_transform(X_w2v)
    elif reduction_method == "umap":
        print(f"Fitting UMAP reduction for neural embeddings ({n_components} components)...")
        import umap
        reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=15, min_dist=0.1)
        X_w2v_reduced = reducer.fit_transform(X_w2v)
    else:
        print("Bypassing dimensionality reduction for neural embeddings...")
        reducer = None
        X_w2v_reduced = X_w2v
        
    # Save reducer
    joblib.dump(reducer, os.path.join(checkpoints_dir, "reducer.joblib"))
    joblib.dump(reduction_method, os.path.join(checkpoints_dir, "reduction_method.joblib"))
    
    # 4. Use Clinical Acoustic Features for Classifier training
    # (We decouple WavLM neural embeddings to prevent OOD bias from corrupting real audio predictions)
    X_fused = X_cli
    print(f"Feature matrix shape for classification: {X_fused.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fused)
    joblib.dump(scaler, os.path.join(checkpoints_dir, "scaler.joblib"))
    
    # Save a copy of training scaled dataset to use as background for SHAP Explainer
    joblib.dump(X_scaled, os.path.join(checkpoints_dir, "background_data.joblib"))
    
    # 5. Define Benchmark Classifiers with Parameter Grids
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.base import clone
    
    grids = {
        'svm': {
            'estimator': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                'kernel': ['rbf', 'linear', 'poly']
            },
            'n_iter': 30
        },
        'random_forest': {
            'estimator': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [4, 6, 8, 10, 12, None],
                'min_samples_split': [2, 3, 5, 8],
                'min_samples_leaf': [1, 2, 3],
                'max_features': ['sqrt', 'log2', 0.2, 0.3, 0.4],
                'criterion': ['gini', 'entropy'],
                'class_weight': ['balanced', None]
            },
            'n_iter': 60
        },
        'logistic_regression': {
            'estimator': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'n_iter': 20
        }
    }
    
    # 6. Benchmarking using 5-Fold Stratified Cross-Validation
    print("\nStarting model benchmarking and hyperparameter optimization...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    benchmarks = {}
    best_tuned_models = {}
    
    for model_name, config in grids.items():
        print(f"Optimizing hyperparameters for {model_name}...")
        search = RandomizedSearchCV(
            estimator=config['estimator'],
            param_distributions=config['params'],
            n_iter=config['n_iter'],
            cv=skf,
            scoring='f1',
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_scaled, y_arr)
        
        best_model = search.best_estimator_
        best_tuned_models[model_name] = best_model
        print(f"Best parameters for {model_name}: {search.best_params_}")
        
        accs, precs, recs, f1s, aucs = [], [], [], [], []
        
        for train_idx, val_idx in skf.split(X_scaled, y_arr):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]
            
            fold_model = clone(best_model)
            fold_model.fit(X_train, y_train)
            
            y_pred = fold_model.predict(X_val)
            y_prob = fold_model.predict_proba(X_val)[:, 1]
            
            accs.append(accuracy_score(y_val, y_pred))
            precs.append(precision_score(y_val, y_pred, zero_division=0))
            recs.append(recall_score(y_val, y_pred, zero_division=0))
            f1s.append(f1_score(y_val, y_pred, zero_division=0))
            aucs.append(roc_auc_score(y_val, y_prob))
            
        benchmarks[model_name] = {
            'accuracy': float(np.mean(accs)),
            'precision': float(np.mean(precs)),
            'recall': float(np.mean(recs)),
            'f1': float(np.mean(f1s)),
            'roc_auc': float(np.mean(aucs))
        }
        
    # 7. Print Benchmark Table in Markdown
    print("\n### Model Performance Comparison")
    print("| Model Class | Accuracy | Precision | Recall | F1-Score | ROC-AUC |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for m_name, metrics in benchmarks.items():
        print(f"| {m_name.upper()} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['roc_auc']:.4f} |")
        
    # Save benchmarks to JSON
    with open(os.path.join(checkpoints_dir, "model_benchmarks.json"), "w") as f:
        json.dump(benchmarks, f, indent=2)
        
    # 8. Select Random Forest as the diagnostic model (Enforce Random Forest)
    best_model_name = 'random_forest'
    print(f"\nSelected Diagnostic Model: **{best_model_name.upper()}** (F1-Score: {benchmarks[best_model_name]['f1']:.4f})")
    
    # Train winning model on all data
    best_model = best_tuned_models[best_model_name]
    best_model.fit(X_scaled, y_arr)
    
    # Save winning checkpoint
    joblib.dump(best_model, os.path.join(checkpoints_dir, "classifier_model.joblib"))
    joblib.dump(best_model_name, os.path.join(checkpoints_dir, "model_type.joblib"))
    print(f"Saved best model checkpoint and type '{best_model_name}' to checkpoints.")
    
    # 9. Build and Save WavLM Reference Assets
    try:
        print("\nGenerating WavLM references...")
        from sklearn.svm import OneClassSVM
        from sklearn.covariance import LedoitWolf
        
        y_arr_np = np.array(y_arr)
        healthy_mask = (y_arr_np == 0)
        parkinsons_mask = (y_arr_np == 1)
        
        healthy_centroid = np.mean(X_w2v, axis=0) if not np.any(healthy_mask) else np.mean(X_w2v[healthy_mask], axis=0)
        parkinsons_centroid = np.mean(X_w2v, axis=0) if not np.any(parkinsons_mask) else np.mean(X_w2v[parkinsons_mask], axis=0)
        
        # Fit OOD Detector (OneClassSVM)
        ood_detector = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
        ood_detector.fit(X_w2v)
        
        # Calculate OOD threshold
        train_scores = ood_detector.score_samples(X_w2v)
        ood_threshold = float(np.percentile(train_scores, 5))
        
        # Reduced embeddings for Mahalanobis
        if reducer is not None:
            X_w2v_reduced = reducer.transform(X_w2v)
        else:
            X_w2v_reduced = X_w2v
            
        mean_healthy_reduced = np.mean(X_w2v_reduced, axis=0) if not np.any(healthy_mask) else np.mean(X_w2v_reduced[healthy_mask], axis=0)
        mean_parkinsons_reduced = np.mean(X_w2v_reduced, axis=0) if not np.any(parkinsons_mask) else np.mean(X_w2v_reduced[parkinsons_mask], axis=0)
        
        # Estimate covariance with Ledoit-Wolf shrinkage
        h_cov_samples = X_w2v_reduced[healthy_mask] if np.any(healthy_mask) else X_w2v_reduced
        p_cov_samples = X_w2v_reduced[parkinsons_mask] if np.any(parkinsons_mask) else X_w2v_reduced
        
        cov_healthy = LedoitWolf().fit(h_cov_samples).covariance_
        cov_parkinsons = LedoitWolf().fit(p_cov_samples).covariance_
        
        # Invert with small regularization
        reg = 1e-4 * np.eye(X_w2v_reduced.shape[1])
        cov_healthy_inv = np.linalg.inv(cov_healthy + reg)
        cov_parkinsons_inv = np.linalg.inv(cov_parkinsons + reg)
        
        references = {
            'healthy_centroid': healthy_centroid,
            'parkinsons_centroid': parkinsons_centroid,
            'train_embeddings': X_w2v,
            'train_labels': y_arr_np,
            'ood_detector': ood_detector,
            'ood_threshold': ood_threshold,
            'mean_healthy_reduced': mean_healthy_reduced,
            'mean_parkinsons_reduced': mean_parkinsons_reduced,
            'cov_healthy_inv': cov_healthy_inv,
            'cov_parkinsons_inv': cov_parkinsons_inv
        }
        
        joblib.dump(references, os.path.join(checkpoints_dir, "wavlm_references.joblib"))
        print("WavLM references generated and saved successfully.")
    except Exception as e:
        print(f"Error generating WavLM references during training: {e}")
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VitaVoice Upgraded Model Training & Benchmarking Pipeline")
    parser.add_argument("--dataset", type=str, default="oxford", choices=["oxford", "real"],
                        help="The dataset type to train on (default: 'oxford')")
    parser.add_argument("--reduction", type=str, default="pca", choices=["pca", "umap", "none"],
                        help="Dimensionality reduction method for neural embeddings (default: 'pca')")
    parser.add_argument("--components", type=int, default=16,
                        help="Number of components for reduction (default: 16)")
    parser.add_argument("--data-dir", type=str, default="datasets",
                        help="Root folder of the datasets (default: 'datasets')")
    args = parser.parse_args()
    
    train_vita_voice(
        dataset_type=args.dataset, 
        root_dir=args.data_dir, 
        reduction_method=args.reduction, 
        n_components=args.components
    )
